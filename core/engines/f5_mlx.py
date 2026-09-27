"""Движок F5-TTS на MLX (нативно на Apple Silicon) — быстрый вариант.

Тот же голос, та же модель и те же ударения, что у torch-движка: берётся ровно
тот же чекпойнт accent_tune, а все шаги подготовки (образец, токенизация текста,
расчёт длительности, громкость, склейка) повторяют torch-версию один в один.
MLX здесь — только другой способ посчитать ту же сеть, быстрее. Сверено: выход
DiT совпадает с torch с точностью float32 (расхождение ~1e-4).

Что было сломано раньше и давало «речеподобную кашу» (три независимые причины):
  1) В кэше модели лежал ФАЙЛ ОТ ДРУГОЙ МОДЕЛИ. Имена слоёв совпадали, поэтому
     он загружался без единой ошибки — но это были не те веса. Теперь на веса
     стоит символьная ссылка, и её цель проверяется при каждой загрузке.
  2) Свой convert_char_to_pinyin в f5-tts-mlx (устаревший) ставит пробел перед
     КАЖДЫМ неlatin-символом: «несмотря» -> «н е с м о т р я». В torch-версии
     эта ветка закрыта проверкой is_chinese(). Здесь перенесён torch-вариант.
  3) Длительность куска считалась своей эвристикой (символы/14) вместо формулы
     F5 (по БАЙТАМ utf-8 и длине образца), из-за чего текст «сжимался» вдвое.
     Плюс образец брался как есть, а torch всегда режет в нём тишину — это ещё
     12% разницы в темпе.

Работает ТОЛЬКО на Apple Silicon (пакеты mlx, f5-tts-mlx, vocos-mlx).
"""

from __future__ import annotations

import os
import re
import sys
import shutil
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
    "ru": {"repo": HF_SRC_REPO, "ckpt": CKPT_REL, "vocab": VOCAB_REL,
           "convert": True,  "stress": "+", "speed_base": 1.0},
    # у английской модели в репозитории тоже лежат веса с ключами «ema_model.*»,
    # то есть в формате PyTorch — конвертировать их надо так же, как русские
    # speed_base=0.62: на общей шкале темпа английская модель читает заметно
    # быстрее русской, и «обычный» темп 1.55 звучал скороговоркой
    "en": {"repo": "lucasnewman/f5-tts-mlx", "ckpt": "model_v1.safetensors",
           "vocab": "vocab.txt", "convert": True, "stress": None,
           "speed_base": 0.62},
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
MAX_TOTAL_SEC = 30
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
}

# Запас времени на кусок — страховка от обрыва последнего слова. Держим его
# небольшим и СРЕЗАЕМ лишнюю тишину после синтеза: без обрезки запас оседал
# хвостами в каждом куске, звук раздувался, а счёт дорожал на треть.
DURATION_HEADROOM = 1.04
SHORT_CHUNK_CHARS = 80        # короче этого куски получают запас побольше
SHORT_HEADROOM = 1.15         # на коротком куске модель торопится сильнее
TRIM_SILENCE_DB = 0.004       # ниже этого уровня хвост считаем тишиной
CFG_STRENGTH = 2.0            # как cfg_strength в f5_tts.infer.utils_infer
SWAY_COEF = -1.0              # как sway_sampling_coef там же


def _mlx_model_dir(lang: str = "ru") -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "text2audio",
                     f"f5mlx_model_{lang}")
    os.makedirs(d, exist_ok=True)
    return d


# --- токенизация: точный перенос f5_tts.model.utils.convert_char_to_pinyin ---
_TRANS = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})


def _is_cjk(c: str) -> bool:
    return "㄀" <= c <= "鿿"


def convert_char_to_tokens(text_list: list[str]) -> list[list[str]]:
    """Текст -> список символов-токенов, как это делает torch-версия F5.

    Для кириллицы результат — просто символы текста (пробелы не вставляются).
    Иероглифы переводятся в пиньинь — ветка оставлена для полноты совместимости.
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
    результат — иначе MLX читает заметно медленнее torch тем же голосом.
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
    """Дописать точку с пробелом в конце — как preprocess_ref_audio_text в torch."""
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text + " " if ref_text.endswith(".") else ref_text + ". "
    return ref_text


def _chunk_by_bytes(text: str, max_bytes: int) -> list[str]:
    """Нарезка куска на под-куски по длине в БАЙТАХ — как chunk_text в torch."""
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


def _trim_tail(wave: np.ndarray) -> np.ndarray:
    """Срезать тишину в конце куска.

    Модели даётся чуть больше времени, чем нужно (страховка от обрыва слова), и
    остаток она заполняет тишиной. Если её не убрать, каждый кусок тащит за
    собой лишний хвост: паузы разъезжаются, а книга становится длиннее без
    всякой пользы. Паузы всё равно ставит пайплайн — ровные и одинаковые.
    """
    if not len(wave):
        return wave
    win = 240                                   # 10 мс при 24 кГц
    n = len(wave) // win
    last = 0
    for i in range(n):
        if np.sqrt(np.mean(wave[i * win:(i + 1) * win] ** 2)) > TRIM_SILENCE_DB:
            last = i
    end = min(len(wave), (last + 1) * win + win * 4)   # оставляем 40 мс на затухание
    return wave[:end]


def _cross_fade(waves: list[np.ndarray], sr: int) -> np.ndarray:
    """Склейка под-кусков с перекрытием — как в infer_batch_process."""
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


class F5MLXEngine(F5Engine):
    name = "f5mlx"
    sample_rate = SAMPLE_RATE
    max_chunk_chars = 200
    supports_cloning = True
    stress_format = "+"

    def __init__(self, voices_dir: str | None = None, model_name: str = HF_SRC_REPO,
                 steps: int = 7, speed: float = 1.55, quant_bits: int | None = None,
                 max_total_sec: float = MAX_TOTAL_SEC, dtype: str = "float32",
                 schedule: str = "epss", language: str = "ru") -> None:
        super().__init__(voices_dir=voices_dir, device=None, nfe_step=steps, speed=speed)
        self.model_name = model_name
        # язык задаёт и модель, и нужны ли ударения: у английской модели свой
        # словарь, знак «+» она прочитала бы как символ
        self.language = language if language in LANG_MODELS else "ru"
        self.stress_format = LANG_MODELS[self.language]["stress"]
        # steps = число вызовов сети. По умолчанию 7 с расписанием EPSS: на целой
        # главе это RTF 0,62 при 94,1% разборчивости против 1,28 и 94,5% у
        # равномерных 16 шагов (эталон torch) — вдвое быстрее при той же речи.
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
        self._audio_cache: dict[str, tuple] = {}
        self._vocoder = None
        self._vocab: dict[str, int] = {}
        # все операции MLX идут через общий поток _MLX_POOL (см. выше)

    # --- подготовка локальной папки модели с ожидаемыми именами файлов ---
    def _prepare_model_dir(self) -> str:
        cfg = LANG_MODELS[self.language]
        """Папка с весами под именем, которого ждёт f5-tts-mlx (model_v1.safetensors).

        Кладём СИМВОЛЬНУЮ ССЫЛКУ на файл из кэша HF и каждый раз проверяем, куда
        она ведёт. Раньше здесь был copy «если файла нет» — и в папке оставался
        файл от прошлых опытов (другая модель с теми же именами слоёв: он
        загружался без ошибок, но голос был не тот и речь выходила кашей).
        """
        from huggingface_hub import hf_hub_download
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

    def load(self, progress=None) -> None:
        _in_mlx_thread(self._load_in_thread)

    def _load_in_thread(self) -> None:
        import f5_tts_mlx.cfm as cfm
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

    # --- образец -> mx-аудио (кэш) --------------------------------------
    def _get_ref(self, voice: str):
        vid, wav, ref_text = self._resolve_voice(voice)
        if vid not in self._audio_cache:
            return _in_mlx_thread(self._make_ref, vid, wav, ref_text)
        return self._audio_cache[vid]

    def _make_ref(self, vid: str, wav: str, ref_text: str):
        import mlx.core as mx
        audio = preprocess_ref_audio(wav)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        gain = 1.0
        if rms < TARGET_RMS:
            gain = TARGET_RMS / rms              # как в torch: поднимаем образец…
            audio = audio * gain
        # …а результат потом опускаем обратно, чтобы громкость совпадала
        self._audio_cache[vid] = (mx.array(audio), _ref_text_for_model(ref_text),
                                  1.0 / gain)
        return self._audio_cache[vid]

    def _max_gen_bytes(self, ref_frames: int, ref_text: str) -> int:
        """Сколько байт текста укладывается в один вызов — формула F5 (torch)."""
        ref_sec = ref_frames * HOP_LENGTH / SAMPLE_RATE
        return max(40, int(len(ref_text.encode("utf-8")) / ref_sec
                           * (self.max_total_sec - ref_sec) * self.speed))

    # --- синтез ---------------------------------------------------------
    def synth_chunk(self, text: str, voice: str, speed_mul: float = 1.0,
                    **kw) -> np.ndarray:
        self.ensure_loaded()
        if not self.has_speech(text):
            return np.zeros(0, dtype=np.float32)

        audio, ref_text, out_gain = self._get_ref(voice)
        ref_frames = audio.shape[0] // HOP_LENGTH
        parts = _chunk_by_bytes(text.strip(), self._max_gen_bytes(ref_frames, ref_text))
        waves: list[np.ndarray] = []
        for part in parts:
            try:
                waves.append(self._synth_one(audio, ref_frames, ref_text, part, speed_mul))
            except Exception as e:
                print(f"[F5-MLX] пропущен кусок: {e}", file=sys.stderr)
        wave = _cross_fade([w for w in waves if len(w)], SAMPLE_RATE)
        if len(wave):
            wave = wave * out_gain
            if not np.all(np.isfinite(wave)):
                wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(wave, dtype=np.float32).reshape(-1)

    def _synth_one(self, audio, ref_frames: int, ref_text: str, gen_text: str,
                   speed_mul: float = 1.0) -> np.ndarray:
        return _in_mlx_thread(self._synth_in_thread, audio, ref_frames,
                              ref_text, gen_text, speed_mul)

    def _synth_in_thread(self, audio, ref_frames: int, ref_text: str,
                         gen_text: str, speed_mul: float = 1.0) -> np.ndarray:
        import mlx.core as mx

        # длительность — формула F5: по длине текста в БАЙТАХ utf-8, плюс запас.
        # Запас нужен против «съеденных» слов: если отведённого времени впритык,
        # модель не успевает договорить и обрывает последнее слово (а иногда
        # комкает первое). Лишнее время она просто заполняет тишиной, которую
        # всё равно съедает пауза между кусками, так что цена запаса нулевая.
        base = LANG_MODELS[self.language].get("speed_base", 1.0)
        local_speed = (0.3 if len(gen_text.encode("utf-8")) < 10
                       else self.speed * speed_mul * base)
        # короткому куску (заголовок, реплика) нужен запас побольше: материала
        # мало, модель торопится и проглатывает последнее слово
        headroom = (SHORT_HEADROOM if len(gen_text) < SHORT_CHUNK_CHARS
                    else DURATION_HEADROOM)
        duration = ref_frames + int(ref_frames / len(ref_text.encode("utf-8"))
                                    * len(gen_text.encode("utf-8")) / local_speed
                                    * headroom)
        tokens = convert_char_to_tokens([ref_text + gen_text])

        mel = self._sample_mel(audio, tokens[0], duration)
        mel = mel[:, ref_frames:, :]          # отбрасываем часть образца (в мел-домене)
        wave = self._vocoder(mel.astype(mx.float32))
        mx.eval(wave)
        return _trim_tail(np.asarray(wave, dtype=np.float32).reshape(-1))

    # --- решение ОДУ (метод Эйлера) с батчевым CFG -----------------------
    def _sample_mel(self, audio, tokens: list[str], duration: int):
        """Тот же расчёт, что F5TTS.sample(method="euler"), но экономнее.

        Отличия только в организации счёта, не в математике:
          * условный и безусловный проходы CFG идут одним батчем (2, n, d) —
            вместо двух отдельных вызовов сети на каждом шаге;
          * эмбеддинг текста и rope считаются один раз на кусок, а не на каждом
            шаге (от шага они не зависят);
          * не копится вся траектория — нужен только последний шаг;
          * длительность — обычный int, поэтому не нужен патч mx.random.normal.
        """
        import mlx.core as mx
        d = self._f5.transformer
        dt = getattr(mx, self.dtype)

        cond = self._f5._mel_spec(audio)                      # (1, n_ref, 100)
        if cond.ndim == 2:
            cond = mx.expand_dims(cond, axis=0)
        ref_len = cond.shape[1]
        n = max(ref_len + 1, int(duration))

        idx = mx.array([[self._vocab.get(c, 0) for c in tokens]], dtype=mx.int32)
        # длина текста не должна превышать длину мела — как lens в F5TTS.sample
        n = max(n, idx.shape[1])

        step_cond = mx.pad(cond, [(0, 0), (0, n - ref_len), (0, 0)]).astype(dt)
        # безусловный проход видит нулевое аудио-условие
        zero_cond = mx.zeros_like(step_cond)

        # текст и rope от шага не зависят — считаем один раз
        te_keep = d.text_embed(idx, n, drop_text=False).astype(dt)
        te_drop = d.text_embed(idx, n, drop_text=True).astype(dt)
        rope = d.rotary_embed.forward_from_seq_len(n)

        x = mx.random.normal((1, n, 100)).astype(dt)

        t = self._timesteps()

        def flow(xc, tc):
            time = mx.broadcast_to(tc.astype(dt), (2,))
            tvec = d.time_embed(time)
            h = mx.concatenate([
                d.input_embed(xc, step_cond, te_keep, drop_audio_cond=False),
                d.input_embed(xc, zero_cond, te_drop, drop_audio_cond=True),
            ], axis=0)
            for blk in d.transformer_blocks:
                h = blk(h, tvec, mask=None, rope=rope)
            out = d.proj_out(d.norm_out(h, tvec))
            pred, null = out[0:1], out[1:2]
            return pred + (pred - null) * CFG_STRENGTH

        flow = mx.compile(flow)
        for i in range(t.shape[0] - 1):
            x = x + (t[i + 1] - t[i]).astype(dt) * flow(x, t[i])
        mx.eval(x)
        return x

    def _timesteps(self):
        """Моменты времени для шагов ОДУ (после sway sampling, как в F5)."""
        import mlx.core as mx
        if self.schedule == "epss":
            sched = EPSS_STEPS.get(self.steps)
            if sched is None:                      # нет такой таблицы — берём ближайшую
                sched = EPSS_STEPS[min(EPSS_STEPS, key=lambda k: abs(k - self.steps))]
            t = mx.array([v / 32.0 for v in sched], dtype=mx.float32)
        else:
            t = mx.linspace(0, 1, self.steps)      # равномерно, как torch-версия
        return t + SWAY_COEF * (mx.cos(mx.pi / 2 * t) - 1 + t)
