"""Движок F5-TTS на MLX (нативно на Apple Silicon) - быстрый вариант.

Тот же голос, та же модель и те же ударения, что у torch-движка: берётся ровно
тот же чекпойнт accent_tune, а все шаги подготовки (образец, токенизация текста,
расчёт длительности, громкость, склейка) повторяют torch-версию один в один.
MLX здесь - только другой способ посчитать ту же сеть, быстрее. Сверено: выход
DiT совпадает с torch с точностью float32 (расхождение ~1e-4).

Что было сломано раньше и давало «речеподобную кашу» (три независимые причины):
  1) В кэше модели лежал ФАЙЛ ОТ ДРУГОЙ МОДЕЛИ. Имена слоёв совпадали, поэтому
     он загружался без единой ошибки - но это были не те веса. Теперь на веса
     стоит символьная ссылка, и её цель проверяется при каждой загрузке.
  2) Свой convert_char_to_pinyin в f5-tts-mlx (устаревший) ставит пробел перед
     КАЖДЫМ неlatin-символом: «несмотря» -> «н е с м о т р я». В torch-версии
     эта ветка закрыта проверкой is_chinese(). Здесь перенесён torch-вариант.
  3) Длительность куска считалась своей эвристикой (символы/14) вместо формулы
     F5 (по БАЙТАМ utf-8 и длине образца), из-за чего текст «сжимался» вдвое.
     Плюс образец брался как есть, а torch всегда режет в нём тишину - это ещё
     12% разницы в темпе.

Работает ТОЛЬКО на Apple Silicon (пакеты mlx, f5-tts-mlx, vocos-mlx).
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import threading

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from .f5 import F5Engine

# ОДИН поток на весь процесс для всех операций MLX.
#
# MLX привязывает и загруженную модель, и скомпилированные через mx.compile
# графы к потоку, в котором они созданы. Кэш компиляции при этом глобальный:
# если второй движок (например, при переключении книги с английской на русскую)
# заведёт свой поток, он достанет из кэша чужой граф и упадёт с
# «There is no Stream(gpu, 1) in current thread», а книга выйдет пустой.
# Поэтому поток должен быть общим и жить столько же, сколько процесс.
_MLX_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def _in_mlx_thread(fn, *args):
    return _MLX_POOL.submit(fn, *args).result()

# Берём ровно ту же accent_tune-модель, что и torch-движок (её голос одобрен).
# Веса в формате PyTorch — f5-tts-mlx сконвертирует их сам (convert_weights=True).
HF_SRC_REPO = "Misha24-10/F5-TTS_RUSSIAN"
CKPT_REL = "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors"
VOCAB_REL = "F5TTS_v1_Base/vocab.txt"

# Модели по языкам. Русская — веса в формате PyTorch, их конвертирует сам
# f5-tts-mlx. Английская уже лежит в MLX-формате, её конвертировать не надо.
LANG_MODELS: dict[str, dict] = {
    # target_bps подобран так, чтобы на «обычном» темпе выходило 26,8 байт/с
    # для русского — ровно та скорость чтения, которую одобрили на слух.
    # У английского байт на символ один, поэтому число вдвое меньше.
    "ru": {"repo": HF_SRC_REPO, "ckpt": CKPT_REL, "vocab": VOCAB_REL,
           "convert": True,  "stress": "+", "target_bps": 29.7},
    # у английской модели в репозитории тоже лежат веса с ключами «ema_model.*»,
    # то есть в формате PyTorch — конвертировать их надо так же, как русские
    "en": {"repo": "lucasnewman/f5-tts-mlx", "ckpt": "model_v1.safetensors",
           "vocab": "vocab.txt", "convert": True, "stress": None,
           "target_bps": 16.1},
}

# Другие веса той же архитектуры F5 По умолчанию -
# Misha24-10 accent_tune (B16, выбран вслепую). ESpeech - запасной (C16), лицензия свободная. ESpeech-TTS-1
# (Apache-2.0) дообучена на русском, понимает «+», словарь тот же; выложена как
# .pt из обучения, поэтому один раз конвертируется в .safetensors в кэш.
MODEL_VARIANTS: dict[str, dict] = {
    "espeech": {"repo": "ESpeech/ESpeech-TTS-1_RL-V2", "pt": "espeech_tts_rlv2.pt",
                "vocab": "vocab.txt"},
}

SAMPLE_RATE = 24000
HOP_LENGTH = 256
TARGET_RMS = 0.1              # как target_rms в f5_tts.infer.utils_infer
CROSS_FADE_SEC = 0.15         # как cross_fade_duration там же
# Сколько секунд (образец + кусок) кладём в ОДИН вызов сети. В torch-версии
# здесь 22 — и наш кусок в 200 символов не влезал, разбивался на два вызова, а
# образец (8.9 с) пересчитывался дважды. При 30 с кусок идёт одним вызовом:
# на этом M4 это RTF 1.40 -> 1.25 при той же разборчивости (проверено ASR).
# F5 v1 обучена на отрезках до ~30 с, так что 8.9 + 15.7 остаётся в пределах.
MAX_TOTAL_SEC = 34
# Расписания шагов EPSS из «Accelerating Flow-Matching-Based TTS via Empirically
# Pruned Step Sampling» (arXiv 2505.19931, Fast F5-TTS). Идея: траектория ОДУ у
# F5 почти вся «решается» в начале, поэтому шаги надо густо ставить у t=0 и
# прореживать к t=1. Это НЕ переобучение и не другая модель — только другие
# моменты времени, поэтому голос остаётся тот же.
# Значения даны в 1/32 доли и потом проходят через sway sampling, как в статье.
# Число вызовов сети = len(расписание) - 1.
EPSS_STEPS: dict[int, tuple[int, ...]] = {
    16: (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32),
    12: (0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32),
    10: (0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
    7:  (0, 2, 4, 6, 8, 16, 24, 32),
    6:  (0, 2, 4, 6, 8, 16, 32),          # вариант 6a из статьи (6b заметно хуже)
    5:  (0, 2, 4, 6, 8, 32),              # у статьи здесь уже разваливается голос
    # 32 - все шаги подряд, как равномерное расписание исходной F5 (эталон,
    # пресет «Высокое»): 33 точки дают ровно 32 вызова сети
    32: tuple(range(33)),
}

# Запас времени на кусок. Если его не дать, модель не укладывается и жертвует
# КРАЯМИ: чаще всего пропадает первое слово («Проба голоса. Море к вечеру…»
# звучит как «Море к вечеру…»), иногда рвётся последнее. Замерено на куске в
# 173 символа: запас 1.04 — первое слово теряется 5 раз из 5, запас 1.22 — ни
# разу (английской модели и этого мало, её страхует проверка в pipeline). Лишнее
# время модель заполняет тишиной; после конца речи от неё остаётся ровно
# TAIL_KEEP_SEC (см. _finish_tail), остальное отбрасывается. Если проверка
# краёв всё же услышала недоговорённое слово, кусок пересчитывается с ещё
# большим запасом (headroom_mul в synth_batch).
DURATION_HEADROOM = 1.22
SHORT_CHUNK_CHARS = 80        # короче этого куски получают запас побольше
SHORT_HEADROOM = 1.30         # на коротком куске модель торопится сильнее
CFG_STRENGTH = 2.0            # как cfg_strength в f5_tts.infer.utils_infer
SWAY_COEF = -1.0              # как sway_sampling_coef там же


def _mlx_model_dir(lang: str = "ru", variant: str | None = None) -> str:
    from ..osutil import model_dir
    d = os.path.join(str(model_dir()), f"f5mlx_model_{lang}" + (f"_{variant}" if variant else ""))
    os.makedirs(d, exist_ok=True)
    return d


# --- токенизация: точный перенос f5_tts.model.utils.convert_char_to_pinyin ---
_TRANS = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})


def _is_cjk(c: str) -> bool:
    return "㄀" <= c <= "鿿"


def convert_char_to_tokens(text_list: list[str]) -> list[list[str]]:
    """Текст -> список символов-токенов, как это делает torch-версия F5.

    Для кириллицы результат - просто символы текста (пробелы не вставляются).
    Иероглифы переводятся в пиньинь - ветка оставлена для полноты совместимости.
    """
    import rjieba
    from pypinyin import lazy_pinyin, Style

    out = []
    for text in text_list:
        chars: list[str] = []
        text = text.translate(_TRANS)
        for seg in rjieba.cut(text):
            seg_byte_len = len(seg.encode("utf-8"))
            if seg_byte_len == len(seg):                      # только ASCII
                if chars and seg_byte_len > 1 and chars[-1] not in " :'\"":
                    chars.append(" ")
                chars.extend(seg)
            elif seg_byte_len == 3 * len(seg):                # только иероглифы
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if _is_cjk(c):
                        chars.append(" ")
                    chars.append(seg_[i])
            else:                                             # смешанный случай
                for c in seg:
                    if ord(c) < 256:
                        chars.extend(c)
                    elif _is_cjk(c):
                        chars.append(" ")
                        chars.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        chars.append(c)                        # кириллица — как есть
        out.append(chars)
    return out


def preprocess_ref_audio(path: str) -> np.ndarray:
    """Подготовка образца ровно как preprocess_ref_audio_text в torch-версии F5.

    torch ВСЕГДА прогоняет образец через вырезание тишины (даже если он короче
    12 с) и дописывает 50 мс тишины в конец. От длины образца напрямую зависит
    расчётная длительность куска, т.е. темп речи, поэтому здесь нужен тот же
    результат - иначе MLX читает заметно медленнее torch тем же голосом.
    """
    from pydub import AudioSegment, silence

    aseg = AudioSegment.from_file(path)

    # 1. режем по длинным паузам, пока не наберётся ~12 с
    segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50,
                                    keep_silence=1000, seek_step=10)
    kept = AudioSegment.silent(duration=0)
    for seg in segs:
        if len(kept) > 6000 and len(kept + seg) > 12000:
            break
        kept += seg

    # 2. если не помогло — по коротким паузам
    if len(kept) > 12000:
        segs = silence.split_on_silence(aseg, min_silence_len=100, silence_thresh=-40,
                                        keep_silence=1000, seek_step=10)
        kept = AudioSegment.silent(duration=0)
        for seg in segs:
            if len(kept) > 6000 and len(kept + seg) > 12000:
                break
            kept += seg

    aseg = kept
    # 3. если тишины для реза не нашлось — просто обрезаем
    if len(aseg) > 12000:
        aseg = aseg[:12000]

    aseg = _remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg = aseg.set_channels(1).set_frame_rate(SAMPLE_RATE)
    a = np.array(aseg.get_array_of_samples(), dtype=np.float32)
    return a / float(1 << (8 * aseg.sample_width - 1))


def _remove_silence_edges(aseg, silence_threshold: int = -42):
    from pydub import silence
    aseg = aseg[silence.detect_leading_silence(aseg, silence_threshold=silence_threshold):]
    tail = silence.detect_leading_silence(aseg.reverse(), silence_threshold=silence_threshold)
    return aseg[: len(aseg) - tail] if tail > 0 else aseg


def _ref_text_for_model(ref_text: str) -> str:
    """Дописать точку с пробелом в конце - как preprocess_ref_audio_text в torch."""
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text + " " if ref_text.endswith(".") else ref_text + ". "
    return ref_text


def voice_weights(voices_dir: str, voice: str) -> str | None:
    """Свои веса голоса (11б, шаг 4: F5, дообученный на записях чтеца): рядом с образцом
    «<голос>.f5model» - строка с папкой, где model_v1.safetensors (формат PyTorch, как у
    Misha) и vocab.txt. Нет файла - веса общие для всех голосов."""
    f = os.path.join(voices_dir, voice + ".f5model")
    if not os.path.isfile(f):
        return None
    d = os.path.expanduser(open(f, encoding="utf-8").read().strip())
    return d if os.path.isfile(os.path.join(d, "model_v1.safetensors")) else None


def _chunk_by_bytes(text: str, max_bytes: int) -> list[str]:
    """Нарезка куска на под-куски по длине в БАЙТАХ - как chunk_text в torch."""
    chunks, cur = [], ""
    for sent in re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text):
        if not sent:
            continue
        tail = sent + " " if len(sent[-1].encode("utf-8")) == 1 else sent
        if len(cur.encode("utf-8")) + len(sent.encode("utf-8")) <= max_bytes:
            cur += tail
        else:
            if cur:
                chunks.append(cur.strip())
            cur = tail
    if cur:
        chunks.append(cur.strip())
    return chunks or [text.strip()]


# Хвост куска (доработки 3.0). Порога громкости здесь больше НЕТ: тихие
# безударные окончания («ждать», «интриг», «выйдет») падали ниже порога и
# отрезались вместе с тишиной. Теперь:
#   * конец речи ищет Silero VAD (а не громкость);
#   * после него ВСЕГДА остаётся TAIL_KEEP_SEC звука модели - затухание слова и
#     немного тишины; если модель столько не дала, хвост дописывается нулями;
#   * в самом конце - короткое затухание FADE_OUT_SEC (против щелчка), оно
#     ложится на тишину, а не на слово.
# Паузу между кусками конвейер потом доводит до заданной (pipeline._pause_fit).
TAIL_KEEP_SEC = 0.25           # звука после конца речи (200-300 мс по заданию)
FADE_OUT_SEC = 0.02            # затухание в самом конце куска: без него - щелчок
FADE_IN_SEC = 0.005            # и короткое нарастание в начале
# Речь, дошедшая до самого конца отведённого времени, - признак, что модели
# не хватило длительности: слово могло оборваться. Такой кусок помечается, и
# конвейер пересчитывает его с бо́льшим запасом.
CUT_RISK_SEC = 0.06
# «Якорь тишины» в конце окна (3.1, лечение съеденных окончаний). Диагноз:
# F5 растягивает речь на ВСЁ отведённое время - сколько ни дай запаса, тишины
# после последнего слова не остаётся (замер: 6 мс при запасе +40 %), и глухая
# согласная на конце («интриг», «ждать») упирается в край окна. Поэтому в
# конец окна кладём ИЗВЕСТНУЮ тишину - как образец кладётся в начало. F5
# обучена дописывать речь между известными кусками, и ей приходится закончить
# слово до тишины, с нормальным затуханием согласной. 0 - выключено (как в 3.0).
TAIL_ANCHOR_SEC = 0.3


def _finish_tail(wave: np.ndarray) -> tuple[np.ndarray, bool]:
    """Хвост куска: речь + ровно TAIL_KEEP_SEC после неё, края сглажены.

    Возвращает (звук, оборвано_ли): второе - True, если речь шла до самого
    конца отведённого времени (модели не хватило длительности).
    """
    if not len(wave):
        return wave, False
    se = None
    try:
        from ..vad import speech_end
        se = speech_end(wave, SAMPLE_RATE)
    except Exception as e:
        from ..log import log_problem
        log_problem("VAD недоступен, хвост куска оставлен целиком", e, show=False)
    cut_risk = se is not None and se >= len(wave) / SAMPLE_RATE - CUT_RISK_SEC
    end = len(wave) if se is None else int((se + TAIL_KEEP_SEC) * SAMPLE_RATE)
    out = wave[:min(end, len(wave))].astype(np.float32, copy=True)
    k = min(len(out), int(FADE_OUT_SEC * SAMPLE_RATE))
    if k > 1:
        out[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32) ** 2
    k = min(len(out), int(FADE_IN_SEC * SAMPLE_RATE))
    if k > 1:
        out[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
    if end > len(out):                       # модель не дала хвоста - дописываем тишину
        out = np.concatenate([out, np.zeros(end - len(out), dtype=np.float32)])
    return out, cut_risk


def _cross_fade(waves: list[np.ndarray], sr: int) -> np.ndarray:
    """Склейка под-кусков с перекрытием - как в infer_batch_process."""
    if not waves:
        return np.zeros(0, dtype=np.float32)
    out = waves[0]
    n = int(CROSS_FADE_SEC * sr)
    for nxt in waves[1:]:
        k = min(n, len(out), len(nxt))
        if k <= 0:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1, 0, k, dtype=np.float32)
        fade_in = np.linspace(0, 1, k, dtype=np.float32)
        mixed = out[-k:] * fade_out + nxt[:k] * fade_in
        out = np.concatenate([out[:-k], mixed, nxt[k:]])
    return out


def _attention_with_mask(self, x, mask=None, rope=None):
    """Attention.__call__ из f5-tts-mlx с исправленной маской.

    В библиотеке ветка с маской не работала никогда: `mask.expand(...)` - у
    массивов MLX такого метода нет (это из torch), и пачка кусков разной длины
    падала. Здесь маска просто расширяется по осям - (B, 1, 1, n) - и дальше
    всё как в оригинале. Без маски путь не меняется ни на одну операцию.
    """
    import mlx.core as mx
    from f5_tts_mlx.dit import apply_rotary_pos_emb
    batch, seq_len, _ = x.shape
    query = self.to_q(x).reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
    key = self.to_k(x).reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
    value = self.to_v(x).reshape(batch, seq_len, self.heads, -1).transpose(0, 2, 1, 3)
    if rope is not None:
        freqs, xpos_scale = rope
        q_s, k_s = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
        query = apply_rotary_pos_emb(query, freqs, q_s)
        key = apply_rotary_pos_emb(key, freqs, k_s)
    attn_mask = mask[:, None, None, :] if mask is not None else None
    x = mx.fast.scaled_dot_product_attention(q=query, k=key, v=value,
                                             scale=self._scale_factor, mask=attn_mask)
    x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1).astype(query.dtype)
    x = self.to_out(x)
    if mask is not None:
        x = x * mask[:, :, None]
    return x


def timesteps(steps: int, schedule: str = "epss") -> np.ndarray:
    """Моменты времени шагов ОДУ после sway sampling - общие для MLX и torch."""
    if schedule == "epss":
        sched = EPSS_STEPS.get(steps)
        if sched is None:                      # нет такой таблицы - берём ближайшую
            sched = EPSS_STEPS[min(EPSS_STEPS, key=lambda k: abs(k - steps))]
        t = np.array([v / 32.0 for v in sched], dtype=np.float32)
    else:
        t = np.linspace(0, 1, steps, dtype=np.float32)   # равномерно, как torch-версия
    return (t + SWAY_COEF * (np.cos(np.pi / 2 * t) - 1 + t)).astype(np.float32)


class F5MLXEngine(F5Engine):
    name = "f5mlx"
    sample_rate = SAMPLE_RATE
    # До 3.3 было 280 знаков: на каждый вызов модель прогоняет образец (около
    # 10 с), и казалось, что длинный кусок выгоднее. Замер 3.3 (одна глава, два
    # зерна, оценка слов по звуку): в длинных кусках модель выкидывает целые
    # фразы («Раз мы здесь» - во всех дублях), а на 2/3 длины слов под
    # подозрением 4 -> 2 и 5 -> 1, и счёт не дольше (RTF 1,50 -> 1,52 и
    # 1,43 -> 1,36: внимание в сети растёт с длиной быстрее, чем линейно).
    max_chunk_chars = 187
    supports_cloning = True
    stress_format = "+"

    def __init__(self, voices_dir: str | None = None, model_name: str = HF_SRC_REPO,
                 steps: int = 16, speed: float = 1.0, quant_bits: int | None = None,
                 max_total_sec: float = MAX_TOTAL_SEC, dtype: str = "float32",
                 schedule: str = "epss", language: str = "ru",
                 variant: str | None = None, cfg_steps: int | None = None) -> None:
        super().__init__(voices_dir=voices_dir, device=None, nfe_step=steps, speed=speed)
        self.model_name = model_name
        # язык задаёт и модель, и нужны ли ударения: у английской модели свой
        # словарь, знак «+» она прочитала бы как символ
        self.language = language if language in LANG_MODELS else "ru"
        self.stress_format = LANG_MODELS[self.language]["stress"]
        # steps = число вызовов сети. По умолчанию 16 с расписанием EPSS (3.0):
        # прежние 7 выбирались по разборчивости ASR (94,1% против 94,5%), но
        # разборчивость - не естественность. В слепом прослушивании 7 шагов дали
        # шум и металл, 16 - лучший голос на повествовании. Цена - вдвое дольше.
        self.steps = steps
        # quant_bits: 8 или 4 — сжатие весов DiT, ускоряет счёт (проверять слух/ASR)
        self.quant_bits = quant_bits
        # сколько секунд (образец + кусок) кладём в один вызов сети: чем больше,
        # тем меньше доля времени на пересчёт образца, но выше нагрузка на модель
        self.max_total_sec = max_total_sec
        # dtype счёта в DiT: float32 (эталон) или float16 (быстрее на видеоядре)
        self.dtype = dtype
        # "epss" — прореженное расписание (то же качество за меньшее число шагов),
        # "uniform" — равномерное, как в torch-версии F5 (эталон для сверки)
        self.schedule = schedule
        if variant and variant not in MODEL_VARIANTS:
            raise ValueError(f"нет варианта весов {variant!r}")
        if variant and language != "ru":
            raise ValueError("другие веса есть только для русского")
        self.variant = variant
        # cfg_steps (3.5, опыт): подсказка CFG (второй проход сети без текста)
        # только на первых шагах; поздние шаги уточняют детали и идут одним
        # проходом - вдвое дешевле. None - на всех шагах, как в F5 (по умолчанию)
        self.cfg_steps = cfg_steps
        self._own_dir: str | None = None   # свои веса голоса (voice_weights) - загружены сейчас
        self._audio_cache: dict[str, tuple] = {}
        self._pace = 1.0                  # поправка темпа под образец
        self._vocoder = None
        self._vocab: dict[str, int] = {}
        # все операции MLX идут через общий поток _MLX_POOL (см. выше)

    # last_cut_risk - своя у каждого потока: упреждение считает следующую пачку,
    # пока основной поток пересчитывает кусок, и общая отметка перезаписывалась
    # посреди чужого вызова (IndexError, поймано test_speed в 3.5)
    @property
    def last_cut_risk(self):
        tls = self.__dict__.setdefault("_tls", threading.local())
        return getattr(tls, "risk", None)

    @last_cut_risk.setter
    def last_cut_risk(self, v):
        tls = self.__dict__.setdefault("_tls", threading.local())
        tls.risk = v

    def cache_key(self) -> str:
        """Всё, что меняет звук куска, кроме текста и голоса (для кэша синтеза).

        TAIL_KEEP_SEC и FADE_OUT_SEC тоже здесь: поменялась обработка хвоста -
        старые куски из кэша не годятся.
        """
        cfg = LANG_MODELS[self.language]
        return "|".join(str(x) for x in (
            self.name, cfg["repo"], cfg["ckpt"], self.variant, self.language, self.steps, self.schedule,
            self.dtype, self.quant_bits, round(self.speed, 4), self.max_total_sec,
            DURATION_HEADROOM, SHORT_HEADROOM, TAIL_KEEP_SEC, FADE_OUT_SEC)
            + ((f"cfg{self.cfg_steps}",) if self.cfg_steps is not None else ()))

    # --- подготовка локальной папки модели с ожидаемыми именами файлов ---
    def _prepare_model_dir(self) -> str:
        cfg = LANG_MODELS[self.language]
        """Папка с весами под именем, которого ждёт f5-tts-mlx (model_v1.safetensors).

        Кладём СИМВОЛЬНУЮ ССЫЛКУ на файл из кэша HF и каждый раз проверяем, куда
        она ведёт. Раньше здесь был copy «если файла нет» - и в папке оставался
        файл от прошлых опытов (другая модель с теми же именами слоёв: он
        загружался без ошибок, но голос был не тот и речь выходила кашей).
        """
        from huggingface_hub import hf_hub_download
        if self._own_dir:
            return self._own_dir
        if self.variant:
            return self._prepare_variant_dir()
        mdir = _mlx_model_dir(self.language)
        src = os.path.realpath(hf_hub_download(cfg["repo"], cfg["ckpt"]))
        dst = os.path.join(mdir, "model_v1.safetensors")
        if not (os.path.islink(dst) and os.path.realpath(dst) == src):
            if os.path.lexists(dst):
                os.chmod(mdir, 0o755)
                os.remove(dst)
            os.symlink(src, dst)
        vocab_src = os.path.realpath(hf_hub_download(cfg["repo"], cfg["vocab"]))
        vocab_dst = os.path.join(mdir, "vocab.txt")
        if not (os.path.islink(vocab_dst) and os.path.realpath(vocab_dst) == vocab_src):
            if os.path.lexists(vocab_dst):
                os.remove(vocab_dst)
            os.symlink(vocab_src, vocab_dst)
        return mdir

    def _prepare_variant_dir(self) -> str:
        """Папка с весами варианта: .pt из обучения -> model_v1.safetensors (разово)."""
        from huggingface_hub import hf_hub_download
        v = MODEL_VARIANTS[self.variant]
        mdir = _mlx_model_dir(self.language, self.variant)
        dst = os.path.join(mdir, "model_v1.safetensors")
        if not os.path.exists(dst):
            import torch
            from safetensors.torch import save_file
            ck = torch.load(hf_hub_download(v["repo"], v["pt"]), map_location="cpu",
                            weights_only=False)
            sd = ck.get("ema_model_state_dict") or ck
            sd = {k: t.contiguous() for k, t in sd.items()
                  if k not in ("initted", "step") and hasattr(t, "contiguous")}
            save_file(sd, dst + ".part")
            os.replace(dst + ".part", dst)
        vocab_src = os.path.realpath(hf_hub_download(v["repo"], v["vocab"]))
        vocab_dst = os.path.join(mdir, "vocab.txt")
        if not (os.path.islink(vocab_dst) and os.path.realpath(vocab_dst) == vocab_src):
            if os.path.lexists(vocab_dst):
                os.remove(vocab_dst)
            os.symlink(vocab_src, vocab_dst)
        return mdir

    def load(self, progress=None) -> None:
        _in_mlx_thread(self._load_in_thread)

    def _load_in_thread(self) -> None:
        import f5_tts_mlx.cfm as cfm
        import f5_tts_mlx.dit as dit
        dit.Attention.__call__ = _attention_with_mask      # см. _attention_with_mask
        from pathlib import Path
        mdir = self._prepare_model_dir()
        # подменяем загрузчик, чтобы взять нашу локальную папку с верным именем файла
        _orig = cfm.fetch_from_hub
        cfm.fetch_from_hub = lambda *a, **k: Path(mdir)
        try:
            # convert_weights=True: веса в формате PyTorch — f5-tts-mlx их конвертирует
            self._f5 = cfm.F5TTS.from_pretrained(
                mdir, convert_weights=LANG_MODELS[self.language]["convert"])
        finally:
            cfm.fetch_from_hub = _orig
        # вокодер отцепляем: сначала обрежем мел-спектр образца, потом озвучим
        # только сгенерированную часть (как делает torch-версия) — это и точнее,
        # и дешевле, чем вокодить образец заново на каждом куске
        self._vocoder = self._f5._vocoder
        self._f5._vocoder = None
        self._vocab = dict(self._f5._vocab_char_map)
        if self.dtype != "float32":
            import mlx.core as mx
            self._f5.transformer.set_dtype(getattr(mx, self.dtype))
            mx.eval(self._f5.parameters())
        if self.quant_bits:
            import mlx.nn as nn
            import mlx.core as mx
            nn.quantize(self._f5, bits=self.quant_bits,
                        class_predicate=lambda p, m: (isinstance(m, nn.Linear)
                                                      and m.weight.shape[1] % 64 == 0))
            mx.eval(self._f5.parameters())
        self._loaded = True

    def _use_weights_of(self, voice: str) -> None:
        """Голос со своими весами (voice_weights) - перезагрузить модель с ними; голос без
        них после такого - вернуть общие. Книга читается одним голосом, так что
        перезагрузка - раз на книгу."""
        vid = os.path.splitext(os.path.basename(voice))[0] if os.path.isfile(voice) else voice
        own = voice_weights(self.voices_dir, vid) if self.language == "ru" else None
        if own == self._own_dir:
            return
        self._own_dir = own
        self._audio_cache.clear()
        if self._loaded:
            self._loaded = False
            self._f5 = None
            self._vocoder = None

    # --- образец -> mx-аудио (кэш) --------------------------------------
    def _get_ref(self, voice: str):
        vid, wav, ref_text = self._resolve_voice(voice)
        if vid not in self._audio_cache:
            return _in_mlx_thread(self._make_ref, vid, wav, ref_text)
        return self._audio_cache[vid]

    def _pace_for(self, audio: np.ndarray, ref_text: str) -> float:
        """Поправка темпа под конкретный образец.

        Выходная скорость речи равна «темп образца × speed». Образцы бывают
        очень разные: у одного диктора в отрывке 29 байт текста на секунду, у
        другого 11 (представляется с паузами). Без поправки один и тот же
        «обычный» темп в окне давал бы то нормальную речь, то тягучую. Поэтому
        приводим всё к целевому темпу языка, а ползунок в окне остаётся
        множителем к нему.
        """
        sec = len(audio) / SAMPLE_RATE
        bps = len(ref_text.encode("utf-8")) / sec if sec else 0
        if bps <= 1:
            return 1.0
        target = LANG_MODELS[self.language].get("target_bps", 27.0)
        return float(np.clip(target / bps, 0.5, 3.5))

    @staticmethod
    def _check_ref_text(vid: str, ref_text: str) -> None:
        """Без расшифровки образца длительность куска не посчитать (вышли бы
        десятки тысяч кадров) - лучше сразу сказать, в чём дело."""
        if len(ref_text.strip()) < 10:
            raise RuntimeError(f"у голоса «{vid}» нет расшифровки образца (.txt) - "
                               "добавьте образец заново")

    def _make_ref(self, vid: str, wav: str, ref_text: str):
        import mlx.core as mx
        self._check_ref_text(vid, ref_text)
        audio = preprocess_ref_audio(wav)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        gain = 1.0
        if rms < TARGET_RMS:
            gain = TARGET_RMS / rms              # как в torch: поднимаем образец…
            audio = audio * gain
        # …а результат потом опускаем обратно, чтобы громкость совпадала
        rt = _ref_text_for_model(ref_text)
        self._audio_cache[vid] = (mx.array(audio), rt, 1.0 / gain,
                                  self._pace_for(audio, rt))
        return self._audio_cache[vid]

    def _max_gen_bytes(self, ref_frames: int, ref_text: str) -> int:
        """Сколько байт текста укладывается в один вызов - формула F5 (torch)."""
        ref_sec = ref_frames * HOP_LENGTH / SAMPLE_RATE
        return max(40, int(len(ref_text.encode("utf-8")) / ref_sec
                           * (self.max_total_sec - ref_sec) * self.speed))

    # --- синтез ---------------------------------------------------------
    def synth_chunk(self, text: str, voice: str, speed_mul: float = 1.0,
                    headroom_mul: float = 1.0, **kw) -> np.ndarray:
        return self.synth_batch([text], voice, [speed_mul], [headroom_mul])[0]

    def synth_batch(self, texts: list[str], voice: str,
                    speed_muls: list[float] | None = None,
                    headroom_muls: list[float] | None = None) -> list[np.ndarray]:
        """Несколько кусков за ОДИН проход сети (пачкой), по звуку на кусок.

        Куски разной длины добиваются до самого длинного, лишнее закрывается
        маской внимания - результат каждого куска тот же, что и по одному
        (проверяется в test_speed.py). headroom_mul > 1 - дать модели ещё
        больше времени (повтор куска с недоговорённым концом).
        """
        self._use_weights_of(voice)
        self.ensure_loaded()
        n = len(texts)
        speed_muls = list(speed_muls or [1.0] * n)
        headroom_muls = list(headroom_muls or [1.0] * n)
        self.last_cut_risk = [False] * n
        audio, ref_text, out_gain, self._pace = self._get_ref(voice)
        ref_frames = audio.shape[0] // HOP_LENGTH
        max_bytes = self._max_gen_bytes(ref_frames, ref_text)
        jobs: list[tuple[int, str]] = []          # (номер куска, под-кусок)
        for i, text in enumerate(texts):
            if self.has_speech(text):
                jobs += [(i, part) for part in _chunk_by_bytes(text.strip(), max_bytes)]
        waves: dict[int, list[np.ndarray]] = {i: [] for i in range(n)}
        if jobs:
            try:
                outs = _in_mlx_thread(
                    self._synth_in_thread, audio, ref_frames, ref_text,
                    [p for _i, p in jobs], [speed_muls[i] for i, _p in jobs],
                    [headroom_muls[i] for i, _p in jobs])
            except Exception as e:
                from ..log import log_problem
                log_problem("[F5-MLX] пачка кусков не озвучилась", e)
                outs = [(np.zeros(0, dtype=np.float32), False)] * len(jobs)
            for (i, _p), (w, risk) in zip(jobs, outs):
                waves[i].append(w)
                self.last_cut_risk[i] = self.last_cut_risk[i] or risk
        result = []
        for i in range(n):
            wave = _cross_fade([w for w in waves[i] if len(w)], SAMPLE_RATE)
            if len(wave):
                wave = wave * out_gain
                if not np.all(np.isfinite(wave)):
                    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
            result.append(np.asarray(wave, dtype=np.float32).reshape(-1))
        return result

    def _gen_frames(self, ref_frames: int, ref_text: str, gen_text: str,
                  speed_mul: float, headroom_mul: float) -> int:
        """Длительность (в кадрах мела) - формула F5: по байтам utf-8, плюс запас.

        Запас нужен против «съеденных» слов: если отведённого времени впритык,
        модель не успевает договорить и обрывает последнее слово (а иногда
        комкает первое). Лишнее время она просто заполняет тишиной.
        """
        local_speed = (0.3 if len(gen_text.encode("utf-8")) < 10
                       else self.speed * speed_mul * self._pace)
        # короткому куску (заголовок, реплика) нужен запас побольше: материала
        # мало, модель торопится и проглатывает последнее слово
        headroom = (SHORT_HEADROOM if len(gen_text) < SHORT_CHUNK_CHARS
                    else DURATION_HEADROOM) * headroom_mul
        return ref_frames + int(ref_frames / len(ref_text.encode("utf-8"))
                                * len(gen_text.encode("utf-8")) / local_speed * headroom)

    def _synth_in_thread(self, audio, ref_frames: int, ref_text: str,
                         gen_texts: list[str], speed_muls: list[float],
                         headroom_muls: list[float]) -> list[tuple[np.ndarray, bool]]:
        import mlx.core as mx
        durs = [self._gen_frames(ref_frames, ref_text, g, sm, hm)
                for g, sm, hm in zip(gen_texts, speed_muls, headroom_muls)]
        tokens = convert_char_to_tokens([ref_text + g for g in gen_texts])
        tail = int(TAIL_ANCHOR_SEC * SAMPLE_RATE / HOP_LENGTH)
        mel, lens = self._sample_mel(audio, tokens, durs, tails=[tail] * len(gen_texts))
        out = []
        for b in range(len(gen_texts)):
            m = mel[b:b + 1, ref_frames:lens[b], :]   # отбрасываем часть образца (в мел-домене)
            wave = self._vocoder(m.astype(mx.float32))
            mx.eval(wave)
            out.append(_finish_tail(np.asarray(wave, dtype=np.float32).reshape(-1)))
        return out

    # --- решение ОДУ (метод Эйлера) с батчевым CFG -----------------------
    def _sample_mel(self, audio, tokens: list[list[str]], durations: list[int],
                    noise: list | None = None, tails: list[int] | None = None):
        """Тот же расчёт, что F5TTS.sample(method="euler"), но экономнее.

        Отличия только в организации счёта, не в математике:
          * условный и безусловный проходы CFG идут одним батчем;
          * несколько кусков идут одним батчем (2 x кусков): разная длина
            добивается до самой большой, добавка закрыта маской - в свёртке
            позиций и во внимании, как в batch-режиме исходной F5;
          * эмбеддинг текста и rope считаются один раз на кусок, а не на каждом
            шаге (от шага они не зависят);
          * не копится вся траектория - нужен только последний шаг.
        noise - начальный шум по куску (для теста: тот же шум по одному и пачкой).
        tails - сколько кадров известной тишины поставить в конец окна куска
        (см. TAIL_ANCHOR_SEC); в результате они заменены самой тишиной.
        Возвращает (мел (B, n, 100), настоящие длины кусков вместе с тишиной).
        """
        import mlx.core as mx
        import mlx.nn as nn
        d = self._f5.transformer
        dt = getattr(mx, self.dtype)
        B = len(tokens)

        cond = self._f5._mel_spec(audio)                      # (1, n_ref, 100)
        if cond.ndim == 2:
            cond = mx.expand_dims(cond, axis=0)
        ref_len = cond.shape[1]
        # длина текста не должна превышать длину мела - как lens в F5TTS.sample
        gen_end = [max(ref_len + 1, int(dur), len(tok)) for dur, tok in zip(durations, tokens)]
        tails = list(tails or [0] * B)
        lens = [g + t for g, t in zip(gen_end, tails)]
        n = max(lens)
        tl = max(len(t) for t in tokens)
        # -1 - «пусто»: внутри text_embed станет заполнителем 0 и закроется маской
        idx = mx.array([[self._vocab.get(c, 0) for c in t] + [-1] * (tl - len(t))
                        for t in tokens], dtype=mx.int32)

        if any(tails):
            # тишина в мел-домене - как у нулевого звука (образец кончается так же)
            sil = self._f5._mel_spec(mx.zeros(((max(tails) + 2) * HOP_LENGTH,)))
            if sil.ndim == 2:
                sil = mx.expand_dims(sil, axis=0)
            rows = []
            for g, t in zip(gen_end, tails):
                parts = [cond, mx.zeros((1, g - ref_len, 100))]
                if t:
                    parts.append(sil[:, :t, :])
                if n - g - t:
                    parts.append(mx.zeros((1, n - g - t, 100)))
                rows.append(mx.concatenate(parts, axis=1))
            step_cond = mx.concatenate(rows, axis=0).astype(dt)
        else:
            step_cond = mx.broadcast_to(
                mx.pad(cond, [(0, 0), (0, n - ref_len), (0, 0)]), (B, n, 100)).astype(dt)
        # безусловный проход видит нулевое аудио-условие
        zero_cond = mx.zeros_like(step_cond)
        same = all(L == n for L in lens)
        mask = None if same else mx.concatenate([mx.arange(n)[None, :] < L for L in lens] * 2,
                                                axis=0)          # (2B, n)

        # текст и rope от шага не зависят - считаем один раз. Каждый кусок -
        # на СВОЕЙ длине, потом добивка нулями: в текстовом эмбеддинге F5 есть
        # глобальная нормировка (GRN) по всей длине, и общая добивка меняла бы
        # результат короткого куска (поймано test_speed)
        if same:
            te_keep = d.text_embed(idx, n, drop_text=False).astype(dt)
            te_drop = d.text_embed(idx, n, drop_text=True).astype(dt)
        else:
            def te(drop):
                rows = []
                for b, L in enumerate(lens):
                    e = d.text_embed(idx[b:b + 1, :len(tokens[b])], L, drop_text=drop)
                    rows.append(mx.pad(e, [(0, 0), (0, n - L), (0, 0)]))
                return mx.concatenate(rows, axis=0).astype(dt)
            te_keep, te_drop = te(False), te(True)
        rope = d.rotary_embed.forward_from_seq_len(n)
        ie = d.input_embed

        conv = ie.conv_pos_embed.conv1d.layers           # Conv1d, Mish, Conv1d, Mish
        m3 = None if mask is None else mask[..., None]

        def embed(xc, c, te):
            # Свёртка позиций с маской после КАЖДОГО слоя. В библиотеке маска
            # стоит только до и после всей пары свёрток, и добивка короткого
            # куска протекала между ними в его последние кадры - туда, где
            # окончание (поймано test_speed: расхождение мела 13 против 0).
            h = ie.proj(mx.concatenate((xc, c, te), axis=-1))
            y = h * m3
            for layer in conv:
                y = layer(y)
                if isinstance(layer, nn.Mish):
                    y = y * m3
            return y + h

        if noise is None:
            x = mx.random.normal((B, n, 100)).astype(dt)
        else:
            x = mx.stack([mx.pad(z[:L], [(0, n - min(L, z.shape[0])), (0, 0)])
                          for z, L in zip(noise, lens)]).astype(dt)
        t = self._timesteps()

        def flow(xc, tc):
            time = mx.broadcast_to(tc.astype(dt), (2 * B,))
            tvec = d.time_embed(time)
            if mask is None:
                h = mx.concatenate([
                    d.input_embed(xc, step_cond, te_keep, drop_audio_cond=False),
                    d.input_embed(xc, zero_cond, te_drop, drop_audio_cond=True),
                ], axis=0)
            else:
                h = embed(mx.concatenate([xc, xc], axis=0), mx.concatenate([step_cond, zero_cond], axis=0),
                          mx.concatenate([te_keep, te_drop], axis=0))
            for blk in d.transformer_blocks:
                h = blk(h, tvec, mask=mask, rope=rope)
            out = d.proj_out(d.norm_out(h, tvec))
            pred, null = out[:B], out[B:]
            return pred + (pred - null) * CFG_STRENGTH

        def flow_cond(xc, tc):
            """Только условный проход (без CFG) - для поздних шагов при cfg_steps."""
            time = mx.broadcast_to(tc.astype(dt), (B,))
            tvec = d.time_embed(time)
            if mask is None:
                h = d.input_embed(xc, step_cond, te_keep, drop_audio_cond=False)
                mk = None
            else:
                mk = mask[:B]
                h = embed_cond(xc)
            for blk in d.transformer_blocks:
                h = blk(h, tvec, mask=mk, rope=rope)
            return d.proj_out(d.norm_out(h, tvec))

        def embed_cond(xc):
            hh = ie.proj(mx.concatenate((xc, step_cond, te_keep), axis=-1))
            y = hh * m3[:B]
            for layer in conv:
                y = layer(y)
                if isinstance(layer, nn.Mish):
                    y = y * m3[:B]
            return y + hh

        flow = mx.compile(flow)
        n_steps = t.shape[0] - 1
        k_cfg = n_steps if self.cfg_steps is None else max(0, min(n_steps, int(self.cfg_steps)))
        if k_cfg < n_steps:
            flow_cond = mx.compile(flow_cond)
        for i in range(n_steps):
            f = flow if i < k_cfg else flow_cond
            x = x + (t[i + 1] - t[i]).astype(dt) * f(x, t[i])
        if any(tails):
            # известные участки - как в F5TTS.sample: на выходе ровно то, что дали
            known = mx.stack([(mx.arange(n) >= g) & (mx.arange(n) < g + t)
                              for g, t in zip(gen_end, tails)], axis=0)[:, :, None]
            x = mx.where(known, step_cond, x)
        mx.eval(x)
        return x, lens

    def _timesteps(self):
        """Моменты времени для шагов ОДУ (после sway sampling, как в F5)."""
        import mlx.core as mx
        return mx.array(timesteps(self.steps, self.schedule))
