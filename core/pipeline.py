"""Оркестровка: файл книги -> аудиокнига, плюс быстрый «пробник».

Синтез глав можно вести параллельно на нескольких ядрах (workers>1) - каждый
рабочий процесс поднимает свой движок и озвучивает свою пачку глав. Подготовку
текста (нормализация, ударения) делаем один раз в родителе.

GUI и CLI вызывают convert_book() (вся книга) или preview_sample() (~1 минута).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .readers import read_document, ScannedPdfError, ReaderError
# CHARS_PER_SEC — только в chapters: две копии однажды разошлись, и прогноз врал
from .chapters import build_chapters, clean_book_title, part_name, CHARS_PER_SEC
from .normalize import normalize_text, sentenize, detect_language
from .stress import add_stress_text, unify_book
from .assemble import (RenderedChapter, write_wav, assemble_m4b, encode_chapter_mp3,
                       polish_wav_to_mp3, concat_to_single, FORMATS, DEFAULT_FORMAT)
from .engines import get_engine, TTSEngine
from .engines.f5 import default_voices_dir
from .voicefx import load as load_fx
from .keepawake import keep_awake
from .log import log_problem, log_info


Progress = Callable[[float, str], None]
MIN_SEC_PER_1000 = 25.0   # меньше этого на 1000 символов — синтез явно не удался

# Модель F5 весит 1,3 ГБ, и грузить её заново на каждое нажатие — это и время,
# и лишняя копия в памяти (окно обычно сначала делает пробник, потом книгу).
_ENGINE_CACHE: dict = {}


def _loaded_engine(engine_id: str, eng_kwargs: dict) -> TTSEngine:
    key = (engine_id, tuple(sorted((eng_kwargs or {}).items())))
    eng = _ENGINE_CACHE.get(key)
    if eng is None:
        eng = get_engine(engine_id, **(eng_kwargs or {}))
        eng.ensure_loaded()
        # держим не больше одной большой модели разом; Silero (40 МБ) - сбоку:
        # «Прочитать вслух» быстрым голосом не должен выгружать клон
        big = engine_id in ("f5mlx", "f5torch", "f5")
        for k in [k for k in _ENGINE_CACHE if (k[0] in ("f5mlx", "f5torch", "f5")) == big
                  and (big or k[0] == engine_id)]:
            del _ENGINE_CACHE[k]
        _ENGINE_CACHE[key] = eng
    return eng


def unload_engines() -> None:
    """Выгрузить модели (сервис после простоя)."""
    _ENGINE_CACHE.clear()
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass                           # не Mac или старый MLX - память вернёт gc


@dataclass
class ConvertOptions:
    make_m4b: bool = False           # единый файл-книга — по желанию
    make_chapter_mp3: bool = True    # по умолчанию отдельные mp3 по главам (потоково)
    minutes_per_file: float = 15.0   # примерная длина одного mp3 по звуку
    hd_filter: bool = True           # полировка звука перед кодированием (см. assemble)
    audio_format: str = DEFAULT_FORMAT
    expand_numbers: bool = True
    use_stress: bool = True
    # Паузы (доработки 3.0, настраиваются в окне): после предложения, после
    # абзаца и между репликами диалога - эта заметно длиннее, на слух это
    # смена говорящего
    gap_sec: float = 0.4
    para_gap_sec: float = 0.8
    dialog_gap_sec: float = 1.1
    workers: int = 1                 # параллельные процессы для синтеза глав
    # Вся книга одним файлом. Синтез всё равно идёт по частям — так первые части
    # можно слушать сразу, а после «Стопа» готовое не пропадает; в конце части
    # склеиваются в один файл, а по отдельности удаляются.
    single_file: bool = False
    use_cache: bool = True           # кэш синтеза по хешу текста (core/synthcache)
    narrator: str = ""               # чтец в тегах (album_artist); пусто - имя голоса
    # Строгая проверка слов всего куска (3.2): пропавшие и неразборчивые слова в
    # середине - пересинтез с другим зерном. Замер: пропавших слов 9 -> 2, но
    # счёт на 50-58 % дольше, поэтому это настройка (по умолчанию включена).
    strict_words: bool = False
    # Проверка слов ПО ЗВУКУ (3.3): текст книги выравнивается по звуку моделью
    # без языковой догадки (core/align); съеденное, смазанное или заменённое
    # слово - пересчёт куска с другим зерном. Заменяет строгую проверку 3.2,
    # которая сравнивала текст Whisper с текстом книги и ловила не то.
    sound_check: bool = True


@dataclass
class ConvertResult:
    m4b_path: Optional[str]
    mp3_dir: Optional[str]
    chapters: int
    duration: float
    title: str
    author: str
    fresh_sec: float = 0.0           # звука посчитано заново (без кэша), 3.8
    cached: int = 0                  # кусков взято из кэша


def _safe_name(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", " ", s).strip()
    return re.sub(r"\s{2,}", " ", s) or "audiobook"


def _prepare_text(text: str, stress_fmt: str | None, expand_numbers: bool,
                  use_stress: bool, language: str = "ru") -> str:
    t = normalize_text(text, expand_numbers=expand_numbers, language=language)
    # ударения ставим только русскому: английская модель прочитала бы «+» вслух
    if use_stress and stress_fmt == "+" and language == "ru":
        t = add_stress_text(t)
    return t


def _split_sentences(text: str, max_chars: int) -> list[tuple[str, bool]]:
    """Предложения абзаца: [(текст, конец_предложения)].

    Слишком длинное предложение режется по запятым и тире; у его внутренних
    кусков второй признак False - это не конец предложения, пауза там короче.
    """
    out: list[tuple[str, bool]] = []
    for sent in sentenize(text):
        if len(sent) > max_chars:
            pieces = _hard_split(sent, max_chars)
            out += [(p, i == len(pieces) - 1) for i, p in enumerate(pieces)]
        else:
            out.append((sent, True))
    return out


def _chunk_sentences(text: str, max_chars: int) -> list[str]:
    return [c for c, _m, _e in _pack_sentences(text, max_chars, 0.0)]


def _pack_sentences(text: str, max_chars: int, gap: float):
    """Предложения абзаца -> куски до max_chars: [(текст, отметки, конец_предл.)].

    Отметки - ((позиция в тексте, пауза), …): где внутри куска кончается
    предложение и какой там должна быть пауза. Модель читает кусок одним
    вызовом и паузы ставит сама; потом конвейер доводит их до заданных
    (_fit_inner_pauses).
    """
    chunks, cur, marks, cur_end = [], "", [], True
    for sent, is_end in _split_sentences(text, max_chars):
        if cur and len(cur) + len(sent) + 1 <= max_chars:
            marks.append((len(cur), gap if cur_end else CLAUSE_GAP))
            cur = cur + " " + sent
        else:
            if cur:
                chunks.append((cur, tuple(marks), cur_end))
            cur, marks = sent, []
        cur_end = is_end
    if cur:
        chunks.append((cur, tuple(marks), cur_end))
    return chunks


def _hard_split(sent: str, max_chars: int) -> list[str]:
    parts, buf = [], ""
    for token in re.split(r"(,|;| — | - )", sent):
        if len(buf) + len(token) <= max_chars:
            buf += token
        else:
            if buf.strip():
                parts.append(buf.strip())
            buf = token
    if buf.strip():
        parts.append(buf.strip())
    out = []
    for p in parts:
        while len(p) > max_chars:
            out.append(p[:max_chars]); p = p[max_chars:]
        if p:
            out.append(p)
    return out


class Cancelled(Exception):
    """Пользователь нажал «Стоп». Уже готовые главы остаются на диске."""


HEADING_SPEED_MUL = 0.88      # заголовок читается медленнее основного текста
# Заголовок, склеенный с первой фразой: замедляем мягче, иначе вместе с
# названием главы растягивается и всё первое предложение.
HEADING_MERGED_MUL = 0.94
# Пауза внутри длинного предложения, разрезанного по запятой (не настраивается)
CLAUSE_GAP = 0.25


def _is_heading(p: str) -> bool:
    """Короткая строка без точки на конце - заголовок (та же мерка, что в chapters)."""
    return (len(p) <= 90 and p[-1] not in ".!?…:,;»\"'"
            and not p[0].islower())


def _is_replica(p: str) -> bool:
    """Абзац - реплика диалога: начинается с тире («— Кто здесь?»)."""
    p = p.lstrip()
    return p[:1] in "—–" or p.startswith("- ")


def _chunk_paragraphs(text: str, max_chars: int, gap: float, para_gap: float,
                      dialog_gap: float | None = None):
    """[(кусок текста, пауза после, множитель темпа, отметки пауз внутри)].

    Три разные паузы (доработки 3.0): после предложения - gap, после абзаца -
    para_gap, между репликами диалога - dialog_gap (по умолчанию заметно
    длиннее: на слух это смена говорящего). Реплика - абзац, начинающийся с
    тире; пауза «между репликами» - когда реплика идёт за репликой.

    Раньше вся глава резалась на куски одной лентой и между ними всегда стояла
    одна и та же пауза 0,35 с, а внутри склеенных кусков пауз не было вовсе.
    """
    if dialog_gap is None:
        dialog_gap = para_gap
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[tuple] = []
    head = ""                      # заголовок ждёт первую фразу главы
    for pi, para in enumerate(paras):
        # Заголовок отправляем в модель ВМЕСТЕ с первым предложением, одним
        # вызовом. Отдельным куском он выходил ненадёжно сразу по двум причинам:
        # кусок короткий (модели не за что зацепиться) и первое слово в куске у
        # F5 нестабильно само по себе. Из-за этого название главы читалось
        # неправильно или пропадало. Вызовов при этом становится МЕНЬШЕ.
        if _is_heading(para):
            head = para if para[-1] in ".!?…" else para + "."
            continue
        nxt = paras[pi + 1] if pi + 1 < len(paras) else ""
        end_gap = dialog_gap if (_is_replica(para) and nxt and _is_replica(nxt)) else para_gap
        parts = _pack_sentences(para, max_chars, gap)
        items = []
        for i, (c, marks, is_end) in enumerate(parts):
            last = i == len(parts) - 1
            items.append([c, end_gap if last else (gap if is_end else CLAUSE_GAP), 1.0, marks])
        if head:
            first = items[0]
            if len(head) + 1 + len(first[0]) <= max_chars:
                shift = len(head) + 1
                first[3] = ((len(head), para_gap),) + tuple((p + shift, g) for p, g in first[3])
                first[0] = head + " " + first[0]
                first[2] = HEADING_MERGED_MUL
            else:                  # первая фраза сама длинная - заголовок отдельно
                out.append((head, para_gap, HEADING_SPEED_MUL, ()))
            head = ""
        out += [tuple(it) for it in items]
    if head:                       # глава из одного заголовка
        out.append((head, para_gap, HEADING_SPEED_MUL, ()))
    return _merge_short(out, max_chars)


# Куски меньше этой доли от максимума склеиваем со следующими. Причина в
# арифметике: на каждый вызов модель заново прогоняет весь образец (около 10 с),
# поэтому кусок на 95 символов стоит почти столько же, сколько на 250, а звука
# даёт втрое меньше. Замерено на книге: куски по 95 символов - синтез 162 с,
# по 200+ - около 110 с при том же объёме звука.
MERGE_UNDER_FRACTION = 0.95
MIN_CHUNK_CHARS = 45
# Заголовок короче этого отдельным куском не оставляем: на 14 символах модели
# не за что зацепиться, и «Проба голоса» превращается в «Проба ГОД».
MIN_HEADING_CHARS = 25


def _merge_short(items, max_chars: int):
    """Склеить короткие куски со следующими.

    На огрызках вроде «Глава 5.» или «Открывая пустыню» F5 ведёт себя плохо:
    материала мало, и модель то глотает слово целиком, то мажет окончание.
    Пауза между склеенными кусками не теряется: она становится отметкой внутри
    куска, и конвейер потом доводит её до заданной (_fit_inner_pauses).
    """
    out: list[tuple] = []
    limit = int(max_chars * MERGE_UNDER_FRACTION)
    for it in items:
        text, gap, mul = it[0], it[1], it[2]
        marks = tuple(it[3]) if len(it) > 3 else ()
        prev_is_heading = out and out[-1][2] != 1.0
        if (out and not prev_is_heading and mul == 1.0
                and len(out[-1][0]) < limit
                and len(out[-1][0]) + len(text) + 2 <= max_chars):
            prev, pgap, pm, pmarks = out[-1]
            sep = " " if prev.endswith((".", "!", "?", "…", ",", ":", ";")) else ". "
            shift = len(prev) + len(sep)
            out[-1] = (prev + sep + text, gap, pm,
                       pmarks + ((len(prev), pgap),) + tuple((p + shift, g) for p, g in marks))
        else:
            out.append((text, gap, mul, marks))
    return out


# --- страховка от съеденных краёв ----------------------------------------
# F5 иногда мажет у самых границ куска: то последнее слово теряет окончание
# («ветер» -> «веть»), то пропадает первое («Море к вечеру стихло» -> «стихло»).
# Каждый синтез стартует со своего случайного шума, поэтому достаточно
# переспросить - со второй попытки кусок обычно выходит чистым. Повтор идёт с
# бо́льшим запасом длительности (HEADROOM_STEPS): если модели не хватило
# времени договорить, второй раз его хватит.
TAIL_RETRIES = 2
HEADROOM_STEPS = (1.0, 1.15, 1.3)
# Модель для проверки краёв. base (130 мс) оказалась слишком слабой: она врала
# на краях («глова» вместо «глава», «гакма» вместо «как море»), и половина
# кусков пересинтезировалась впустую. small - 310 мс и уже не путается, против
# 1260 мс у large-v3-turbo. Сравниваем слова строго на конце (окончание должно
# совпасть: «правил» против «правило» - брак) и нечётко в начале.
_ASR_MODEL = "small"          # core/asr: MLX на Mac, faster-whisper на Windows/Linux
# Строгая проверка ВСЕГО куска (3.2): слово в середине пропало или искажено
# («и огляделся», неразборчивое «вернётся»). Тут small мало - он сам путает
# слова, - нужен large-v3-turbo. Сверка по последним буквам, как для окончаний:
# распознаватель «дописывает» обрубленное по смыслу (урок 3.1).
_ASR_FULL = "turbo"
STRICT_WORDS = False          # 3.3: выключена - на слух выигрыша не дала, а время +50 %
CLONE_ENGINES = ("f5mlx", "f5torch", "f5")   # темп - параметр движка только у клона
SOUND_CHECK = True            # 3.3: проверка слов по звуку (core/align)
EDGES_BY_SOUND = True         # 3.5: края куска у клона - по звуку, без Whisper
WORD_RETRIES = 2
STRESS_RETRIES = 2
# Проверка ударения по звуку у слов с риском (3.8, core/stresscheck): ударение
# не там - пересчёт куска с другим зерном. По умолчанию ВЫКЛЮЧЕНА: задание 10а
# заменило этот путь (модель, которая слушается ударений); на слух не проверена.
# HERALD_STRESS_CHECK=1 - включить для опытов.
STRESS_CHECK = os.environ.get("HERALD_STRESS_CHECK", "0") == "1"
STRESS_TAG = " (ударение)"
# Движок, который слушается ударений (Qwen3-TTS ru-stress, 3.8 / 10а): готовый
# кусок проверяет meter2 (core/meter); ударение не там - пересчёт, не больше
# STRESS_RETRIES раз. HERALD_METER_CHECK=0 - выключить.
METER_CHECK = os.environ.get("HERALD_METER_CHECK", "1") != "0"
_align_failed = False


def _sound_words(audio: np.ndarray, sr: int, text: str):
    """Оценки слов по звуку или None, если модель выравнивания недоступна."""
    if _align_failed:
        return None
    try:
        from .align import aligner
        return aligner().score(audio, sr, text)
    except Exception as e:
        _sound_failed(e)
        return None
# END_PUNCT: кусок без знака препинания в конце получает точку - без неё F5
# чаще недоговаривает последнее слово. False - прежнее поведение (для замера).
END_PUNCT = True
_END_MARKS = ".!?…;:,»\")—-"
_WORD_SIMILAR = 0.75
_asr_failed = False
# Сколько кусков считать за один проход сети (см. F5MLXEngine.synth_batch).
# Замер на M4 (docs/Приёмка 3.1, «Скорость»), 12 кусков главы без переозвучек:
# по одному - RTF 1,80, пачкой по 2 - 1,97 (медленнее: видеоядро и так занято,
# а добивка до длинного куска - лишний счёт). Поэтому по одному; пачки
# оставлены рабочими (результат совпадает до бита, test_speed) на случай
# другого железа.
BATCH_SIZE = 1


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w ]", " ", s.lower().replace("+", "").replace("ё", "е")).split()


def _last_word(s: str) -> str:
    w = _words(s)
    return w[-1] if w else ""


def _sounds_like(want: str, heard: list[str]) -> bool:
    """Есть ли среди услышанных слов похожее на нужное."""
    import difflib
    return any(difflib.SequenceMatcher(None, want, w).ratio() >= _WORD_SIMILAR
               for w in heard)


def _ends_like(want: str, heard: list[str]) -> bool:
    """Прозвучало ли последнее слово ЦЕЛИКОМ - сверяем его КОНЕЦ (3.1).

    Сравнивать слово целиком мало: распознаватель по смыслу «дописывает»
    недоговорённое («интри» -> «интриг»), и схожесть выходит высокой. Поэтому
    требуем совпадения последних трёх букв (двух у слов короче пяти) - там и
    живут съеденные глухие «г», «т», «ть». Смотрим два последних услышанных
    слова: распознаватель иногда добавляет лишнее в конце.
    """
    import difflib
    k = 3 if len(want) >= 5 else 2
    # до двух лишних слов после последнего: на тишине в конце Whisper дописывает
    # («продолжение следует») - замер 3.2
    for h in heard[-4:]:
        if h == want:
            return True
        if (h[-k:] == want[-k:]
                and difflib.SequenceMatcher(None, want, h).ratio() >= 0.6):
            return True
    return False


def _with_end_punct(text: str) -> str:
    if END_PUNCT and text.strip() and text.rstrip()[-1] not in _END_MARKS:
        return text.rstrip() + "."
    return text


def _listen(audio: np.ndarray, sr: int, lang: str = "ru",
            model: str | None = None) -> list[tuple] | None:
    """Распознать кусок целиком: [(слово, начало с, конец с)] или None.

    Одно распознавание на кусок даёт сразу и проверку краёв, и места пауз
    между предложениями (по меткам времени слов).
    """
    global _asr_failed
    if _asr_failed or len(audio) < sr // 2:
        return None
    se = None
    try:
        # хвостовую тишину не даём: на ней Whisper сочиняет («продолжение
        # следует») - и проверка последнего слова видит чужое слово
        from .vad import speech_end
        se = speech_end(audio, sr)
        if se is not None:
            audio = audio[:min(len(audio), int((se + 0.2) * sr))]
    except Exception:
        se = None                           # без VAD - распознаём целиком
    try:
        import tempfile as _tf
        import soundfile as _sf
        from . import asr
        with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _sf.write(f.name, audio, sr)
        try:
            # язык передаём явно: иначе whisper определяет его сам, и это
            # ровно удваивает время проверки
            r = asr.transcribe(f.name, model or _ASR_MODEL, language=lang,
                               word_timestamps=True)
        finally:
            os.unlink(f.name)
        out = []
        for seg in r.get("segments", []):
            for w in seg.get("words", []):
                # слово, начатое уже в тишине после речи, - выдумка распознавателя
                if se is not None and float(w["start"]) > se + 0.05:
                    continue
                for ww in _words(w.get("word", "")):
                    out.append((ww, float(w["start"]), float(w["end"])))
        return out
    except Exception as e:
        _asr_failed = True                  # нет модели - дальше не проверяем
        # без этой проверки съеденные края слов уже не перезаписываются
        log_problem("проверка краёв кусков (распознавание) недоступна, "
                    "съеденные окончания не будут переозвучиваться", e)
        return None


def _edges_verdict(heard: list[tuple] | None, text: str, audio_sec: float) -> bool:
    """Слышны ли первое и последнее слова куска - те самые, что в тексте."""
    if heard is None or audio_sec < 1.0:
        return True
    want = _words(text)
    got = [w for w, _s, _e in heard]
    if not want or not got:
        return bool(not want)
    # слова короче четырёх букв распознаватель путает сам - их не проверяем
    if len(want[0]) >= 4 and not _sounds_like(want[0], got[:2]):
        return False
    if len(want[-1]) >= 4:
        # многоточие в конце - речь обрывается по тексту («Не выйде…»): слово
        # не дописано самим автором, по буквам его не сверить - только «похоже»
        trailing = text.rstrip().endswith(("...", "…"))
        ok = (_sounds_like(want[-1], got[-2:]) if trailing
              else _ends_like(want[-1], got))
        if not ok:
            return False
    return True


_known_cache: dict = {}


def _known_word(w: str) -> bool:
    """Слово из словаря RUAccent (3,2 млн форм)? Выдуманные имена книги
    («Экстервитом») распознаватель пишет как попало - их не сверяем, иначе
    каждый кусок с таким именем пересчитывался бы впустую."""
    if w in _known_cache:
        return _known_cache[w]
    try:
        from .stress import _get_accentizer
        acc = _get_accentizer()
        ok = w in acc.accents or w in acc.omographs
    except Exception:
        ok = True
    _known_cache[w] = ok
    return ok


def _bad_words(heard: list[tuple] | None, text: str) -> list[str]:
    """Слова текста, которых нет в распознанном или у которых другой конец.

    Сверяем только слова из словаря длиной от четырёх букв: короткие
    распознаватель путает сам. Выравнивание - по последовательности слов
    (difflib); в заменённом участке слова сопоставляются по порядку, и слово
    засчитано, только если у него те же последние буквы (3, у коротких 2).
    """
    if heard is None:
        return []
    import difflib
    want = _words(text)
    # те же слова, но с «ё» - словарь RUAccent хранит их так
    raw = re.sub(r"[^\w ]", " ", text.lower().replace("+", "")).split()
    if len(raw) != len(want):
        raw = want
    got = [w for w, _s, _e in heard]
    bad = []
    sm = difflib.SequenceMatcher(None, want, got, autojunk=False)
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal":
            continue
        for k, w in enumerate(want[a0:a1]):
            if len(w) < 4 or not _known_word(raw[a0 + k]):
                continue
            h = got[b0 + k] if tag == "replace" and b0 + k < b1 else ""
            if not (h and _same_word(w, h)):
                bad.append(w)
    return bad


_VOWELS = set("аеёиоуыэюяй")


def _same_word(w: str, h: str) -> bool:
    """Слово в СЕРЕДИНЕ куска прозвучало? (3.2, по замеру ложных срабатываний)

    Распознаватель путает безударные окончания («носителя» -> «носители»,
    «серией нажатий» -> «серии нажатия»): это не дефект синтеза, а его
    слух. Поэтому засчитываем, если слово очень похоже (>= 0,8) или у слова
    тот же согласный «скелет» - гласные (и «й») не различаются.
    Пропавшее слово (нет пары вовсе) и невнятное (мало похоже) - не засчитано.
    """
    import difflib
    r = difflib.SequenceMatcher(None, w, h).ratio()
    if r >= 0.8:
        return True
    return r >= 0.6 and _skeleton(w) == _skeleton(h)


def _skeleton(x: str) -> str:
    """Согласный «скелет» слова: подряд идущие гласные (и «й») - одна «*»."""
    out = []
    for c in x:
        c = "*" if c in _VOWELS else c
        if not (c == "*" and out and out[-1] == "*"):
            out.append(c)
    return "".join(out)


def _edges_ok(audio: np.ndarray, sr: int, text: str, head_only: bool = False,
              lang: str = "ru") -> bool:
    """Совместимость: проверка краёв одного куска."""
    return _edges_verdict(_listen(audio, sr, lang), text, len(audio) / sr)


# --- паузы ----------------------------------------------------------------
# Паузу ищем как участок тишины: кадры по 10 мс тише SIL_REL от громкости речи
# куска. Слово при этом не трогается: от конца речи до правки остаётся
# KEEP_AFTER, до начала следующего слова - KEEP_BEFORE. Это не обрезка по
# порогу: звук только раздвигается или сдвигается ВНУТРИ тишины.
SIL_FRAME = 0.01
SIL_REL = 0.03             # -30 дБ от громкости речи
# Короче - это не пауза, а стык слов или смычка согласной внутри слова («г» в
# «угла» - 50-120 мс тишины): растянуть такое - испортить слово. Не трогаем.
MIN_INNER_SIL = 0.15
KEEP_AFTER = 0.12
KEEP_BEFORE = 0.06
XFADE = 0.008
LONG_PAUSE = 0.5           # абзац и смена говорящего - паузу ставим, даже если модель её не дала
DIP_REL = 0.35             # ...но только в провал громкости между словами (-9 дБ от речи)


def _frame_rms(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float, int]:
    """(громкость кадров по 10 мс, громкость речи куска, размер кадра)."""
    win = max(1, int(SIL_FRAME * sr))
    n = len(audio) // win
    if n == 0:
        return np.zeros(0), 0.0, win
    rms = np.sqrt(np.mean(audio[:n * win].reshape(n, win) ** 2, axis=1))
    return rms, float(np.percentile(rms, 90)), win


def _silent_frames(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    rms, level, win = _frame_rms(audio, sr)
    return rms < max(1e-4, level * SIL_REL), win


def edge_silence(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(тишина в начале, тишина в конце) куска, секунды."""
    sil, win = _silent_frames(audio, sr)
    if not len(sil) or sil.all():
        return 0.0, len(audio) / sr
    loud = np.where(~sil)[0]
    return loud[0] * win / sr, (len(audio) - (loud[-1] + 1) * win) / sr


def _set_pause(audio: np.ndarray, sr: int, start: int, end: int, target: float) -> np.ndarray:
    """Тишину audio[start:end] сделать длиной target секунд (правка внутри неё)."""
    L = end - start
    T = int(target * sr)
    ka, kb = int(KEEP_AFTER * sr), int(KEEP_BEFORE * sr)
    p = start + min(ka, L // 2)
    xf = int(XFADE * sr)
    if T >= L:                                  # раздвинуть: вставить тишину
        ins = np.zeros(T - L, dtype=np.float32)
        a, b = audio[:p].copy(), audio[p:].copy()
        k = min(xf, len(a), len(b))
        if k > 1:
            a[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
            b[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        return np.concatenate([a, ins, b])
    cut = min(L - T, max(0, end - kb - p))      # сжать: вырезать середину тишины
    if cut <= 0:
        return audio
    a, b = audio[:p].copy(), audio[p + cut:].copy()
    k = min(xf, len(a), len(b))
    if k > 1:
        mix = a[-k:] * np.linspace(1, 0, k, dtype=np.float32) + b[:k] * np.linspace(0, 1, k, dtype=np.float32)
        return np.concatenate([a[:-k], mix, b[k:]])
    return np.concatenate([a, b])


def _word_spans(text: str) -> list[tuple[str, int]]:
    """Слова текста с позицией конца (для привязки отметок пауз к словам)."""
    out = []
    for m in re.finditer(r"[\w+]+", text):
        for w in _words(m.group(0)):
            out.append((w, m.end()))
    return out


def _fit_inner_pauses(audio: np.ndarray, sr: int, text: str, marks,
                      heard: list[tuple] | None) -> tuple[np.ndarray, int]:
    """Паузы между предложениями внутри куска - довести до заданных.

    Модель читает кусок (2-4 предложения, иногда реплики разных героев) одним
    вызовом и паузы ставит сама. Место каждой паузы находим по меткам времени
    распознанных слов: конец последнего слова предложения и начало первого
    слова следующего. Внутри этого промежутка берём самый длинный участок
    тишины и делаем его нужной длины. Если слова не сопоставились или тишины
    нет - пауза остаётся, как её поставила модель (ничего не ломаем).
    Возвращает (звук, сколько пауз поправлено).
    """
    if not marks or not heard:
        return audio, 0
    import difflib
    spans = _word_spans(text)
    tw = [w for w, _e in spans]
    hw = [w for w, _s, _e in heard]
    sm = difflib.SequenceMatcher(None, tw, hw, autojunk=False)
    t2h = {}
    for blk in sm.get_matching_blocks():
        # одиночное совпадение («не», «и») могло встать не туда, если модель
        # пропустила или исказила слова рядом; доверяем только связкам из двух+
        if blk.size < 2:
            continue
        for k in range(blk.size):
            t2h[blk.a + k] = blk.b + k
    sil, win = _silent_frames(audio, sr)
    rms, level, _w = _frame_rms(audio, sr)
    edits = []
    for pos, target in marks:
        a_idx = max((i for i, (_w, e) in enumerate(spans) if e <= pos), default=None)
        if a_idx is None or a_idx + 1 >= len(spans):
            continue
        b_idx = a_idx + 1
        # слово по краям паузы должно быть узнано; соседнее - допускаем, но не дальше
        left = next((t2h[i] for i in (a_idx, a_idx - 1) if i in t2h), None)
        right = next((t2h[i] for i in (b_idx, b_idx + 1) if i in t2h), None)
        if left is None or right is None or right <= left:
            continue
        # ищем ПОСЛЕ конца слова (метки Whisper неточны на сотые - небольшой запас)
        lo = int(max(0.0, heard[left][2] - 0.05) / SIL_FRAME)
        hi = int(min(len(audio) / sr, heard[right][1] + 0.1) / SIL_FRAME)
        best, cur = None, None
        for f in range(max(0, lo), min(len(sil), hi)):
            if sil[f]:
                cur = (cur[0], f + 1) if cur else (f, f + 1)
                if best is None or cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
            else:
                cur = None
        if best is None or (best[1] - best[0]) * SIL_FRAME < MIN_INNER_SIL:
            # Модель склеила абзацы или реплики вовсе без паузы («…немедленно!
            # - Не могу»). Для длинной паузы (абзац, смена говорящего) тишину
            # вставляем в самой тихой точке между двумя словами - но только
            # если там правда провал громкости, а не середина звука.
            if target < LONG_PAUSE:
                continue
            a0 = int(heard[left][2] / SIL_FRAME) - 2
            b0 = int(heard[right][1] / SIL_FRAME) + 2
            a0, b0 = max(0, a0), min(len(rms), b0)
            if b0 - a0 < 1:
                continue
            f = a0 + int(np.argmin(rms[a0:b0]))
            if rms[f] > level * DIP_REL:
                continue
            best = (f, f + 1)
        edits.append((best[0] * win, best[1] * win, target))
    for start, end, target in sorted(edits, reverse=True):   # с конца: начала не сдвигаются
        audio = _set_pause(audio, sr, start, end, target)
    return audio, len(edits)


def _join(pieces: list[tuple[np.ndarray, float]], sr: int) -> np.ndarray:
    """Склеить куски так, чтобы пауза между ними была ровно заданной.

    Пауза = тишина в конце куска (его хвост, см. f5_mlx._finish_tail) +
    вставка + тишина в начале следующего. Вставляем ровно столько, сколько
    не хватает до заданной; после последнего куска - его пауза целиком.
    """
    j = _Joiner(sr)
    for audio, gap in pieces:
        j.add(audio, gap)
    j.finish()
    return j.result()


def _drop(fut) -> None:
    """Брошенный заказ упреждения: отменить, а если уже считается - дождаться,
    чтобы движок освободился раньше, чем его возьмёт следующий заказ."""
    if fut is not None and not fut.cancel():
        try:
            fut.result()
        except Exception:
            pass                       # результат всё равно не нужен


class _Joiner:
    """Та же склейка, что _join, но по одному куску (звук потоком, core/stream).

    Пауза после куска зависит от тишины в начале СЛЕДУЮЩЕГО, поэтому она
    уходит вместе со следующим куском: [пауза перед ним, он сам].
    """

    def __init__(self, sr: int, emit=None):
        self.sr, self.emit = sr, emit
        self.out: list[np.ndarray] = []
        self._prev = None                       # (хвост тишины, пауза) прошлого куска

    def _gap(self, nxt_lead: float) -> np.ndarray:
        trail, gap = self._prev
        return np.zeros(int(self.sr * max(0.0, gap - trail - nxt_lead)), dtype=np.float32)

    def add(self, audio: np.ndarray, gap: float, cost: float | None = None) -> None:
        lead, trail = edge_silence(audio, self.sr)
        seg = [audio] if self._prev is None else [self._gap(lead), audio]
        self._prev = (trail, gap)
        self._put(seg, len(audio) / self.sr, cost)

    def finish(self) -> None:
        if self._prev is not None:
            self._put([self._gap(0.0)], 0.0, None)
            self._prev = None

    def _put(self, seg: list, chunk_sec: float, cost: float | None) -> None:
        self.out += seg
        if self.emit is not None:
            self.emit(np.concatenate(seg), chunk_sec, cost)

    def result(self) -> np.ndarray:
        return (np.concatenate(self.out) if self.out
                else np.zeros(int(self.sr * 0.2), dtype=np.float32))


def _synth_by_sentences(eng, voice: str, text: str, mul: float, sr: int,
                        gap: float) -> np.ndarray:
    """Кусок по предложениям: каждое - отдельным вызовом, склейка с паузой.

    Лечение стойкого пропуска фразы в длинном куске (3.3). Дороже обычного
    пересчёта (образец прогоняется на каждое предложение), поэтому - только
    когда пересчёт с другим зерном не помог.
    """
    sents = [s for s, _end in _split_sentences(text, 280)] or [text]
    if len(sents) < 2:
        return eng.synth_batch([text], voice, [mul])[0]
    waves = eng.synth_batch([_with_end_punct(s) for s in sents], voice, [mul] * len(sents))
    return _join([(w, gap) for w in waves if len(w)], sr)


def _norm_items(chunks, gap_sec: float) -> list[tuple]:
    """Строки, пары (текст, пауза), тройки (…, темп) и четвёрки (…, отметки)."""
    items = []
    for c in chunks:
        if isinstance(c, str):
            items.append((c, gap_sec, 1.0, ()))
        elif len(c) == 2:
            items.append((c[0], c[1], 1.0, ()))
        elif len(c) == 3:
            items.append((c[0], c[1], c[2], ()))
        else:
            items.append((c[0], c[1], c[2], tuple(tuple(m) for m in c[3])))
    return items


# Упреждение: пока окно проверяет края готовых кусков (распознавание) и
# доводит паузы, видеоядро уже считает следующую пачку
_PREFETCH = None


def _prefetch_pool():
    global _PREFETCH
    if _PREFETCH is None:
        from concurrent.futures import ThreadPoolExecutor
        _PREFETCH = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    return _PREFETCH


def _synth_chunks(eng: TTSEngine, voice: str, chunks, sr: int,
                  gap_sec: float, on_chunk=None, should_stop=None,
                  lang: str = "ru", pause_gate=None, batch_size: int | None = None,
                  use_cache: bool = True, stats: dict | None = None,
                  pipelined: bool = True, strict_words: bool | None = None,
                  sound_check: bool | None = None, on_piece=None,
                  live=None, check_edges: bool = True) -> np.ndarray:
    """Куски -> звук части: пачками, с упреждением, кэшем и проверкой краёв.

    stats (если передан) пополняется: fresh_sec - звука посчитано заново,
    cached - кусков из кэша, retries - переозвучек, pauses_fit - пауз внутри
    кусков доведено до заданных.
    """
    from . import synthcache
    if strict_words is None:
        strict_words = STRICT_WORDS
    if sound_check is None:
        sound_check = SOUND_CHECK
    # по звуку проверяется только клон F5: готовые голоса Silero слов не
    # глотают, а выравнивание шло бы дольше их собственного счёта
    sound_check = sound_check and _clone(eng)
    items = _norm_items(chunks, gap_sec)
    texts = [_with_end_punct(t) for t, _g, _m, _k in items]
    # звук потоком (core/stream): куски уходят в игру по мере готовности, склейка
    # та же, что у файла; «Стоп» потока останавливает и синтез
    joiner = _Joiner(sr, live.feed if live is not None else None)
    if live is not None:
        live.plan([(t, g) for t, g, _m, _k in items], sr)
        _stop0 = should_stop
        should_stop = lambda: live.should_stop() or bool(_stop0 and _stop0())  # noqa: E731
    base_muls = [m for _t, _g, m, _k in items]

    class _Muls:
        """Темп куска: свой (заголовок медленнее) × темп потока. Темп потока
        можно менять на ходу (ползунок во вкладке «Прочитать вслух») - он
        действует с ещё не посчитанного куска."""
        def __getitem__(self, i):
            return base_muls[i] * (getattr(live, "tempo", 1.0) if live is not None else 1.0)
    muls = _Muls()
    # края у клона - по звуку (3.5): тем же выравниванием, что слова. Оба судьи
    # сразу (Whisper И звук) на «Пробе пера» дали 18 вызовов вместо 14 и RTF 2,02
    # против 1,41: каждый лишний пересчёт - ~10 с счёта модели
    sound_edges = (EDGES_BY_SOUND and check_edges and lang == "ru" and not strict_words
                   and _clone(eng))
    st = stats if stats is not None else {}
    for k in ("fresh_sec", "cached", "retries", "pauses_fit"):
        st.setdefault(k, 0)
    B = max(1, int(batch_size or BATCH_SIZE))
    groups = [list(range(i, min(i + B, len(items)))) for i in range(0, len(items), B)]
    vdir = default_voices_dir()
    ekey = eng.cache_key()
    used_mul: dict[int, float] = {}
    stress_on = STRESS_CHECK and sound_check and lang == "ru" and _clone(eng)
    meter_on = (METER_CHECK and lang == "ru" and getattr(eng, "name", "") == "mlxaudio"
                and getattr(eng, "model", "") == "qwen-stress")
    if stress_on:
        from . import stresscheck as SC
        hard = SC.hard_words(ekey)

    def key_of(i, m=None):
        return synthcache.key(ekey, vdir, voice, texts[i], muls[i] if m is None else m)

    def produce(g):
        """Звук пачки: из кэша или синтезом. Работает в потоке упреждения."""
        out = {}
        todo = []
        for i in g:
            used_mul[i] = muls[i]
            hit = synthcache.get(key_of(i, used_mul[i])) if use_cache else None
            if hit is not None:
                out[i] = (hit[0], hit[1], True, False)
            else:
                todo.append(i)
        if todo:
            t0 = time.monotonic()
            waves = eng.synth_batch([texts[i] for i in todo], voice, [muls[i] for i in todo])
            _spent(todo, t0)
            risks = list(getattr(eng, "last_cut_risk", None) or [False] * len(todo))
            for i, w, r in zip(todo, waves, risks):
                out[i] = (w, None, False, r)
        return out

    # сколько стоил каждый кусок: счёт, проверки, пересчёты (звук потоком меряет
    # по этому скорость; интервал между кусками при упреждении обманчив)
    cost: dict[int, float] = {}

    def _spent(idx, t0):
        dt = (time.monotonic() - t0) / max(1, len(idx))
        for i in idx:
            cost[i] = cost.get(i, 0.0) + dt

    pool = _prefetch_pool()
    fut = None
    for gi, g in enumerate(groups):
        # «Пауза»: ждём здесь, между пачками, не выходя из процесса - модель
        # остаётся загруженной, готовые куски на месте, продолжение мгновенное
        if pause_gate is not None:
            pause_gate()
        if should_stop is not None and should_stop():
            _drop(fut)
            raise Cancelled()
        got = fut.result() if fut is not None else produce(g)
        fut = None
        model = _ASR_FULL if strict_words else _ASR_MODEL
        # Края - распознаванием ДО заказа следующего куска (3.5): Whisper и F5
        # на Mac делят видеоядро, и одновременно проверка тянулась 8 с вместо
        # 0,3 с (замер «Проба пера»: 136 с из 251). Так она идёт одна, а
        # выравнивание (процессор) - уже параллельно со счётом следующего.
        pre_h: dict[int, list | None] = {}
        for i in g:
            a_, _w, cached_, _r = got[i]
            if not cached_ and len(a_) and check_edges and not sound_edges:
                t0 = time.monotonic()
                pre_h[i] = _listen(a_, sr, lang, model)
                _spent([i], t0)
        # после «Стопа» следующий кусок не заказываем: движок нужен новому заказу
        # Поток (3.4): первый кусок проверяется БЕЗ упреждения. Иначе распознавание
        # его краёв делит видеоядро с синтезом второго и тянется 9 с вместо 0,3 -
        # первый звук приходил только вместе со вторым куском (замер 3.4)
        first_live = live is not None and gi == 0
        if (pipelined and gi + 1 < len(groups) and not first_live
                and not (should_stop and should_stop())):
            fut = pool.submit(produce, groups[gi + 1])
        heard: dict[int, list | None] = {}

        def judge(i, audio, risk, h=None):
            """(годится, оценка): оценка - чем больше, тем лучше дубль.

            По звуку оценка дубля - его САМОЕ ПЛОХОЕ слово (а не среднее): из
            пересчётов берётся тот, где худшее слово звучит лучше всего."""
            # Края - по звуку, если звук и так выравнивается (3.5): Whisper на
            # видеоядре делил его с синтезом и на Mac тянулся 8 с на кусок вместо
            # 0,3 с (замер «Проба пера»: 136 с из 251). Без проверки слов, при
            # строгой проверке 3.2 и если выравнивание недоступно - Whisper.
            # выравнивание нужно и для краёв: у клона - всегда, даже без проверки
            # слов (иначе без неё снова Whisper на видеоядре - медленнее, чем с ней)
            ws = (_sound_words(audio, sr, texts[i])
                  if len(audio) and lang == "ru" and (sound_check or sound_edges) else None)
            lp = None
            if ws and stress_on:
                from .align import aligner as _al
                lp = getattr(_al(), "last_lp", None)
            if sound_edges and ws:
                from .align import edges_ok
                edges = not risk and (len(audio) / sr < 1.0 or edges_ok(ws, texts[i]))
            else:
                if h is None and len(audio) and check_edges:
                    h = _listen(audio, sr, lang, model)  # пересчёт - распознаём здесь
                edges = not risk and _edges_verdict(h, texts[i], len(audio) / sr)
            bad = _bad_words(h, texts[i]) if strict_words else []
            if ws and not strict_words:
                # времена слов по выравниванию точнее, чем у Whisper: по ним и паузы
                # внутри куска (слова те же, что в тексте)
                h = [(x, w.start, w.end) for w in ws for x in _words(w.word)]
            worst = 0.0
            if sound_check and len(audio) and lang == "ru":
                if ws:
                    from .align import is_bad, stretched
                    bad = bad + [w.word for w in ws if is_bad(w)]
                    # крайне растянутая гласная - тоже повод пересчитать (3.4)
                    bad = bad + [w + " (тянется)" for w in stretched(ws, audio, sr)]
                    worst = min(w.post_mean for w in ws)
                    if lp is not None:
                        # ударение по звуку у редких слов, омографов и трудных (3.8)
                        try:
                            wrong = SC.wrong_words(ws, lp, texts[i], _al(), hard,
                                                   [SC.NEW_REFS_PER_CHUNK])
                        except Exception as e:
                            log_problem("проверка ударения по звуку не сработала", e, show=False)
                            wrong = []
                        bad = bad + [w + STRESS_TAG for w in wrong]
            if meter_on and len(audio):
                try:
                    from . import meter
                    bad = bad + [w + STRESS_TAG for w in meter.wrong_words(audio, sr, texts[i])]
                except Exception as e:
                    log_problem("meter2 не проверил ударения", e, show=False)
            if bad or not edges:
                heard_txt = " ".join(w for w, _s, _e in (h or []))
                log_info(f"проверка куска: края {'да' if edges else 'НЕТ'}, слова не так: "
                         f"{bad} | слова: {heard_txt[:300]}")
            # из дублей: края целы, потом без растянутых гласных, потом лучшее худшее слово
            long_ = sum(1 for b in bad if b.endswith(" (тянется)"))
            n_stress = sum(1 for b in bad if b.endswith(STRESS_TAG))
            return h, edges and not bad, (edges, -long_, -n_stress, worst, -len(bad)), edges, bad

        best: dict[int, tuple] = {}
        need = []
        for i in g:
            audio, words, cached, risk = got[i]
            if cached:
                st["cached"] += 1
                heard[i] = words
                continue
            if not len(audio):
                heard[i] = None
                continue
            t0 = time.monotonic()
            h, ok, score, edges, bad = judge(i, audio, risk, pre_h.get(i))
            _spent([i], t0)
            heard[i] = h
            best[i] = (score, audio, h, bad)
            if not ok:
                need.append((i, edges))
        tried = set()
        # стойкий пропуск слова - по словам, не по ударениям: ударение чинит зерно,
        # а чтение по предложениям тут ни при чём
        first_bad = {i: {b for b in best[i][3] if not b.endswith(STRESS_TAG)} for i, _e in need}
        last_bad = dict(first_bad)
        for attempt in range(1, max(TAIL_RETRIES, WORD_RETRIES if (strict_words or sound_check)
                                    else 0, STRESS_RETRIES if meter_on else 0) + 1):
            if not need:
                break
            st["retries"] += len(need)
            tried.update(i for i, _e in need)
            # Стойкий пропуск (3.3): если после пересчёта с другим зерном под
            # подозрением ТЕ ЖЕ слова, третий раз зерно не поможет - модель
            # выкидывает фразу из длинного куска («Раз мы здесь» пропадало во
            # всех дублях). Тогда кусок читается по предложениям, каждое
            # отдельно, и склеивается с обычными паузами.
            split = [i for i, e in need if sound_check and e and attempt >= 2
                     and last_bad.get(i, set()) & first_bad.get(i, set())]
            reseed = [(i, e) for i, e in need if i not in split]
            # край съеден - даём больше времени; слово в середине - просто
            # другое зерно (каждый синтез стартует со своего шума)
            hms = [HEADROOM_STEPS[min(attempt, len(HEADROOM_STEPS) - 1)] if not e else 1.0
                   for _i, e in reseed]
            idx = [i for i, _e in reseed]
            t0 = time.monotonic()
            waves = (eng.synth_batch([texts[i] for i in idx], voice, [muls[i] for i in idx], hms)
                     if idx else [])
            _spent(idx, t0)
            risks = list(getattr(eng, "last_cut_risk", None) or [False] * len(idx))
            for i in split:
                idx.append(i)
                t0 = time.monotonic()
                waves.append(_synth_by_sentences(eng, voice, texts[i], muls[i], sr, gap_sec))
                _spent([i], t0)
                risks.append(False)
                st["split"] = st.get("split", 0) + 1
            still = []
            for i, w, r in zip(idx, waves, risks):
                if not len(w):
                    continue
                t0 = time.monotonic()
                h, ok, score, edges, bad = judge(i, w, r)
                _spent([i], t0)
                last_bad[i] = {b for b in bad if not b.endswith(STRESS_TAG)}
                if score > best[i][0]:
                    best[i] = (score, w, h, bad)
                only_stress = bad and edges and all(b.endswith(STRESS_TAG) for b in bad)
                if only_stress:
                    st["stress_retries"] = st.get("stress_retries", 0) + 1
                if not ok and not (only_stress and attempt >= STRESS_RETRIES):
                    still.append((i, edges))
            need = still
        st["retry_chunks"] = st.get("retry_chunks", 0) + len(tried)
        if stress_on:
            # и после пересчётов ударение не там - слово трудное для этой модели:
            # дальше оно проверяется всегда (список пополняется сам)
            stubborn = [SC._plain(b[:-len(STRESS_TAG)])
                        for _s, _w, _h, bad in best.values() for b in bad if b.endswith(STRESS_TAG)]
            if stubborn:
                SC.remember_hard(ekey, stubborn)
                st["stress_bad"] = st.get("stress_bad", 0) + len(stubborn)
        for i, (score, w, h, bad) in best.items():
            got[i] = (w, None, False, False)
            heard[i] = h
            st["bad_words"] = st.get("bad_words", 0) + len(bad)
            if bad:
                st.setdefault("bad_list", []).extend(bad)
        for i in g:
            audio, _w, cached, _r = got[i]
            if not cached and len(audio):
                st["fresh_sec"] += len(audio) / sr
                if use_cache:                        # лучший из дублей - окончательный
                    synthcache.put(key_of(i, used_mul.get(i)), audio, sr, heard[i])
            if on_piece is not None and len(audio):
                on_piece(texts[i], audio)          # стенды: звук куска до пауз
            if len(audio):
                audio, nfit = _fit_inner_pauses(audio, sr, texts[i], items[i][3], heard[i])
                st["pauses_fit"] += nfit
                joiner.add(audio, items[i][1], cost.get(i))
            if on_chunk:
                on_chunk()
            if live is not None and live.should_stop():
                _drop(fut)
                raise Cancelled()
    joiner.finish()
    if live is not None:
        live.finish()
    return joiner.result()


# --- параллельные рабочие процессы --------------------------------------
_W: dict = {}


def _worker_init(engine_id: str, eng_kwargs: dict, threads: int) -> None:
    try:
        import torch
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass                      # число потоков — только скорость, не результат
    eng = get_engine(engine_id, **(eng_kwargs or {}))
    eng.ensure_loaded()
    _W["eng"] = eng


def _worker_render(payload: tuple) -> tuple:
    ci, title, chunks, voice, sr, gap_sec, _para_gap, workdir = payload
    eng = _W["eng"]
    audio = _synth_chunks(eng, voice, chunks, sr, gap_sec)
    wav_path = os.path.join(workdir, f"ch{ci:04d}.wav")
    dur = write_wav(audio, sr, wav_path)
    return ci, wav_path, dur, title


# --- продолжение после сбоя ---------------------------------------------
# Сырые части книги лежат не во временной папке, а в скрытой «.herald» рядом с
# готовыми файлами — в FLAC, вдвое меньше WAV. Рядом state.json: для каждой
# части отпечаток «текст после ударений + голос + настройки». Если книгу
# прервали (закрыли окно, сбой, «Стоп»), следующий запуск пропускает части с
# тем же отпечатком. Книга досчиталась — «.herald» удаляется целиком.
STATE_DIR = ".herald"
STATE_FILE = "state.json"
STATE_VERSION = 2               # 2: паузы и хвосты кусков (доработки 3.0)


def settings_fingerprint(engine: str, eng_kwargs: dict, voice: str,
                         opts: "ConvertOptions") -> str:
    """Отпечаток всего, что меняет звук, кроме самого текста части."""
    import hashlib
    vdir = default_voices_dir()
    vinfo = {}
    from .voicefx import fx_path
    for ext in (".wav", ".txt", ".fx.json"):
        p = fx_path(vdir, voice) if ext == ".fx.json" else os.path.join(vdir, voice + ext)
        if os.path.exists(p):
            with open(p, "rb") as f:
                vinfo[ext] = hashlib.sha1(f.read()).hexdigest()
    data = {
        "v": STATE_VERSION, "engine": engine,
        "kw": {k: eng_kwargs[k] for k in sorted(eng_kwargs)},
        "voice": voice, "voice_files": vinfo,
        "opts": {k: getattr(opts, k) for k in (
            "minutes_per_file", "hd_filter", "audio_format", "expand_numbers",
            "use_stress", "gap_sec", "para_gap_sec", "dialog_gap_sec",
            "single_file", "strict_words", "sound_check")},
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                   default=str).encode("utf-8")).hexdigest()


def _part_hash(fp: str, chunks) -> str:
    import hashlib
    body = json.dumps([list(c) if not isinstance(c, str) else c for c in chunks],
                      ensure_ascii=False)
    return hashlib.sha1((fp + body).encode("utf-8")).hexdigest()


def _load_state(state_dir: str) -> dict:
    p = os.path.join(state_dir, STATE_FILE)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        log_problem("файл продолжения озвучки повреждён, начинаю книгу заново", e)
        return {}


def _save_state(state_dir: str, state: dict) -> None:
    p = os.path.join(state_dir, STATE_FILE)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)              # атомарно: сбой посреди записи не портит файл


SERVICE_DIR = "_служебное"
READ_TEXT = "текст с ударениями.txt"
DISPUTED_FILE = "спорные ударения.txt"      # RUAccent и silero-stress разошлись (3.8)
CHUNK_LIST = "куски.json"           # 3.8: по нему прогноз видит, что уже в кэше


def save_read_text(book_dir: str, prepared) -> str | None:
    """Как будет прочитано (3.7): текст каждой части ровно в том виде, в каком он
    уйдёт в модель - с числами словами и ударениями («+» перед ударной гласной).
    По нему видно, дошло ли ударение до синтеза. Лежит в _служебное/ рядом с
    частями; окно показывает его по кнопке «Как будет прочитано»."""
    try:
        d = os.path.join(book_dir, SERVICE_DIR)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, READ_TEXT)
        with open(path, "w", encoding="utf-8") as f:
            for ci, (title, chunks) in enumerate(prepared, 1):
                f.write(f"=== Часть {ci}{'. ' + title if title else ''} ===\n\n")
                for c in chunks:
                    f.write(_with_end_punct(c[0]) + "\n")
                f.write("\n")
        return path
    except OSError as e:
        log_problem("не записался «текст с ударениями»", e, show=False)
        return None


def save_disputed(book_dir: str, words) -> None:
    try:
        d = os.path.join(book_dir, SERVICE_DIR)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, DISPUTED_FILE), "w", encoding="utf-8") as f:
            f.write("\n".join(words) + ("\n" if words else ""))
    except OSError as e:
        log_problem("не записались спорные ударения", e, show=False)


def read_disputed(path: str, out_dir: str | None = None) -> list[str]:
    """Спорные слова книги (после озвучки или «Как будет прочитано»)."""
    try:
        p = os.path.join(output_dir_for(path, out_dir), SERVICE_DIR, DISPUTED_FILE)
        return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    except OSError:
        return []


def prepare_read_text(path: str, opts: "ConvertOptions") -> str:
    """«Как будет прочитано» без озвучки: та же подготовка, что в convert_book.
    Спорные ударения (RUAccent против silero-stress) - в _служебное/."""
    from .stress import take_disputed
    take_disputed()
    doc = read_document(path)
    chapters = build_chapters(doc, opts.minutes_per_file)
    lang = detect_language(doc.text or " ".join(c.text for c in chapters[:3]))
    bodies = [_prepare_text(ch.text, "+", opts.expand_numbers, opts.use_stress, lang)
              for ch in chapters]
    if opts.use_stress and lang == "ru":
        bodies, _u = unify_book(bodies)
    out = []
    for ci, (ch, body) in enumerate(zip(chapters, bodies), 1):
        chunks = _chunk_paragraphs(body, 187, opts.gap_sec, opts.para_gap_sec, opts.dialog_gap_sec)
        out.append(f"=== Часть {ci}{'. ' + ch.title if ch.title else ''} ===\n\n"
                   + "\n".join(_with_end_punct(c[0]) for c in chunks) + "\n")
    try:
        save_disputed(output_dir_for(path), take_disputed())
    except Exception as e:
        log_problem("спорные ударения не сохранились", e, show=False)
    return "\n".join(out)


def read_text_path(path: str, out_dir: str | None = None) -> str:
    return os.path.join(output_dir_for(path, out_dir), SERVICE_DIR, READ_TEXT)


def _calibrate_stress(eng, voice: str, lang: str, say, should_stop=None) -> None:
    """Новая модель (установка / обновление): один раз прогнать тестовый набор
    ударений - какие слова клон читает не так (core/stresscheck). ~3 мин."""
    if not (STRESS_CHECK and lang == "ru"
            and os.environ.get("HERALD_STRESS_CALIBRATE", "1") != "0"):
        return
    try:
        from . import stresscheck as SC
        from .align import aligner
        from .engines import accel_label
        if SC.calibrated(eng.cache_key()) or accel_label() == "cpu":
            return                    # на процессоре набор шёл бы четверть часа
        say(0.05, "Проверяю, как модель ставит ударения (один раз для модели, ~3 мин)…")
        from .stream import ENGINE_LOCK
        with ENGINE_LOCK:
            bad = SC.calibrate(eng, voice, aligner(), should_stop=should_stop)
        log_info(f"тестовый набор ударений: трудные для модели - {', '.join(bad) or 'нет'}")
    except Exception as e:
        log_problem("тестовый набор ударений не прогнался", e, show=False)


def save_chunk_list(book_dir: str, prepared, gap_sec: float, src: str) -> None:
    """Куски книги ровно как они уйдут в модель (текст и темп) - для прогноза
    (3.8: «Проба пера 3» второй раз - прогноз 2:19, факт 0:55, куски были в кэше)."""
    try:
        items = []
        for ci, (_title, chunks) in enumerate(prepared, 1):
            for t, _g, m, _k in _norm_items(chunks, gap_sec):
                items.append([ci, _with_end_punct(t), m])
        st = os.stat(src)
        d = os.path.join(book_dir, SERVICE_DIR)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, CHUNK_LIST), "w", encoding="utf-8") as f:
            json.dump({"book": [st.st_size, st.st_mtime_ns], "chunks": items}, f,
                      ensure_ascii=False)
    except OSError as e:
        log_problem("список кусков для прогноза не записался", e, show=False)


def cached_share(path: str, *, engine: str, eng_kwargs: dict, voice: str,
                 language: str = "ru", out_dir: str | None = None) -> dict | None:
    """Сколько кусков этой книги уже в кэше синтеза при ТЕКУЩИХ настройках.

    {"n": кусков, "cached": из них в кэше, "chars": букв, "cached_chars": ...}
    или None, если книгу ещё не озвучивали или файл книги поменялся. Ключи
    кэша считаются заново (движок, голос, темп) - другой голос или режим
    честно даст «ничего не готово»."""
    from . import synthcache
    try:
        man = os.path.join(output_dir_for(path, out_dir), SERVICE_DIR, CHUNK_LIST)
        data = json.load(open(man, encoding="utf-8"))
        st = os.stat(path)
        if data.get("book") != [st.st_size, st.st_mtime_ns]:
            return None
        kw = dict(eng_kwargs or {})
        if engine in CLONE_ENGINES:
            kw.setdefault("language", language)
            vs = load_fx(default_voices_dir(), voice).speed      # как в convert_book
            if vs and vs != 1.0:
                kw["speed"] = kw.get("speed", 1.0) * vs
        ekey = get_engine(engine, **kw).cache_key()
        vdir = default_voices_dir()
        out = {"n": 0, "cached": 0, "chars": 0, "cached_chars": 0}
        for _ci, text, mul in data.get("chunks", []):
            hit = os.path.exists(synthcache._path(synthcache.key(ekey, vdir, voice, text, mul)))
            out["n"] += 1
            out["chars"] += len(text)
            out["cached"] += hit
            out["cached_chars"] += len(text) if hit else 0
        return out if out["n"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def output_dir_for(path: str, out_dir: str | None = None) -> str:
    """Куда ляжет озвучка книги (та же логика, что в convert_book)."""
    doc = read_document(path)
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    return os.path.join(out_dir, _safe_name(clean_book_title(doc.title or doc.base_title)))


def resume_info(path: str, *, engine: str, eng_kwargs: dict, voice: str,
                options: "ConvertOptions", language: str | None = None,
                out_dir: str | None = None) -> dict | None:
    """Есть ли незаконченная озвучка этой книги с ТЕМИ ЖЕ настройками.

    Возвращает {"done": готово частей, "total": всего} или None. Сам текст здесь
    не проверяется (ударения на всю книгу считаются минутами) - это делает
    convert_book по каждой части перед тем, как её пропустить.
    """
    try:
        sd = os.path.join(output_dir_for(path, out_dir), STATE_DIR)
        st = _load_state(sd)
        if not st or st.get("finished"):
            return None
        kw = dict(eng_kwargs)
        if engine in ("f5mlx", "f5", "f5torch") and "language" not in kw and language:
            kw["language"] = language
        if st.get("fingerprint") != settings_fingerprint(engine, kw, voice, options):
            return None
        done = sum(1 for d in st.get("parts", {}).values()
                   if os.path.exists(os.path.join(sd, d.get("flac", ""))))
        return {"done": done, "total": int(st.get("total", 0))} if done else None
    except Exception as e:
        log_problem("не удалось проверить, есть ли незаконченная озвучка", e, show=False)
        return None


def forget_progress(path: str, out_dir: str | None = None) -> None:
    """«Начать заново»: стереть сохранённые части незаконченной озвучки."""
    shutil.rmtree(os.path.join(output_dir_for(path, out_dir), STATE_DIR),
                  ignore_errors=True)


def cleanup_stale_temp(max_age_hours: float = 24.0) -> int:
    """Удалить забытые временные папки программы старше суток.

    Остаются после сбоев и после прежних версий, где временная папка с WAV
    глав не удалялась вовсе - на восьмичасовой книге это гигабайты за раз.
    """
    import time as _time
    root = tempfile.gettempdir()
    now, n = _time.time(), 0
    for name in os.listdir(root):
        if not name.startswith(("t2a_", "herald_fx_")):
            continue
        p = os.path.join(root, name)
        try:
            if now - os.path.getmtime(p) < max_age_hours * 3600:
                continue
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            n += 1
        except OSError as e:
            log_problem(f"не удалось удалить старый временный файл {p}", e, show=False)
    if n:
        log_info(f"удалено старых временных файлов и папок: {n}")
    try:
        from . import synthcache
        k = synthcache.prune()
        if k:
            log_info(f"кэш синтеза: удалено старых кусков {k}")
    except Exception as e:
        log_problem("кэш синтеза не почистился", e, show=False)
    return n


def _model_msg(engine: str, kw: dict | None) -> str:
    """«Готовлю модель…» - а в первый раз честно: идёт скачивание моделей."""
    try:
        if engine in ("f5mlx", "f5torch") and not (kw or {}).get("variant"):
            from huggingface_hub import try_to_load_from_cache
            from .engines.f5_mlx import LANG_MODELS
            cfg = LANG_MODELS.get((kw or {}).get("language", "ru"), LANG_MODELS["ru"])
            if not isinstance(try_to_load_from_cache(cfg["repo"], cfg["ckpt"]), str):
                return ("Скачиваю модели (первый запуск, около 4,5 ГБ - это может занять "
                        "полчаса)…")
    except Exception:
        pass                          # не узнали - просто обычная надпись
    return "Готовлю модель синтеза…"


def _clone(eng) -> bool:
    return str(getattr(eng, "name", "")).startswith("f5")


def _prepare_sound_check(say, kw: dict | None) -> None:
    """Модель проверки слов по звуку - загрузить до первого куска. Иначе после
    обновления с 3.2 первый кусок молча ждал бы скачивания 1,3 ГБ."""
    if (kw or {}).get("language", "ru") != "ru" or _align_failed:
        return
    try:
        from huggingface_hub import try_to_load_from_cache
        from .align import MODEL, aligner
        if not isinstance(try_to_load_from_cache(MODEL, "config.json"), str):
            say(0.05, "Скачиваю модель проверки слов (один раз, 1,3 ГБ)…")
        aligner().load()
    except Exception as e:
        _sound_failed(e)


def _sound_failed(e: Exception) -> None:
    global _align_failed
    _align_failed = True
    log_problem("проверка слов по звуку недоступна, куски не перепроверяются", e)


# --- вся книга -----------------------------------------------------------
def convert_book(path: str, *, engine: str = "f5", eng_kwargs: dict | None = None,
                 voice: str, out_dir: str | None = None,
                 options: ConvertOptions | None = None,
                 progress: Progress | None = None,
                 on_chapter: Callable | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 pause_gate: Callable[[], None] | None = None,
                 part_progress: Callable[[int, int, float], None] | None = None,
                 on_rtf: Callable[[float], None] | None = None,
                 ) -> ConvertResult:
    """Озвучить книгу. Главы выдаются потоково: как только глава готова, её mp3
    сразу пишется рядом и вызывается on_chapter(index, title, mp3_path) - можно
    начинать слушать, пока остальные считаются. В конце собирается единый .m4b.

    pause_gate() зовётся перед каждым куском и может ждать сколько угодно -
    это «Пауза». part_progress(k, m, доля) сообщает, как далеко озвучена
    текущая часть k из m (общий прогресс по книге идёт через progress).
    on_rtf(rtf) - настоящий RTF: секунды счёта на секунду НОВОГО звука, без
    пауз пользователя и без кусков из кэша. Зовётся после каждой части.
    """
    opts = options or ConvertOptions()
    eng_kwargs = eng_kwargs or {}
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    # язык книги выбирает и модель синтеза, и нужны ли ударения
    doc_lang = detect_language(read_document(path).text or "")
    if engine in ("f5mlx", "f5", "f5torch") and "language" not in eng_kwargs:
        eng_kwargs = dict(eng_kwargs, language=doc_lang)
    meta = get_engine(engine, **eng_kwargs)      # без загрузки модели: только метаданные
    sr = meta.sample_rate
    stress_fmt = getattr(meta, "stress_format", None)

    def say(fr, msg):
        if progress:
            progress(max(0.0, min(1.0, fr)), msg)

    say(0.0, "Читаю файл…")
    doc = read_document(path)
    chapters = build_chapters(doc, opts.minutes_per_file)
    if not chapters:
        raise ReaderError("Не удалось извлечь текст из файла.")

    say(0.04, "Готовлю текст (числа, ударения)…")
    from .stress import take_disputed
    take_disputed()
    # заголовок уже стоит первой строкой в ch.text (см. chapters.py), отдельно
    # его добавлять нельзя — иначе название прочитается дважды
    bodies = [_prepare_text(ch.text, stress_fmt, opts.expand_numbers,
                            opts.use_stress, doc_lang) for ch in chapters]
    if opts.use_stress and stress_fmt == "+" and doc_lang == "ru":
        # имя героя должно звучать одинаково во всей книге (см. stress.unify_book)
        bodies, unified = unify_book(bodies)
        if unified:
            log_info("единое ударение по книге: " + ", ".join(
                f"{v} ({n})" for v, n in list(unified.values())[:40]))
            print(f"ℹ Ударения: у {len(unified)} слов вне словаря выровнено по книге "
                  f"({sum(n for _w, n in unified.values())} мест)", file=sys.stderr)
    prepared = [(ch.title, _chunk_paragraphs(body, meta.max_chunk_chars, opts.gap_sec,
                                             opts.para_gap_sec, opts.dialog_gap_sec))
                for ch, body in zip(chapters, bodies)]

    # настройки голоса: тембр применяем при кодировании, темп — при синтезе.
    # Сбой здесь меняет звук (книга выйдет без поправок тембра), поэтому —
    # в журнал и в окно, а не молча.
    fx_chain = ""
    try:
        fx_chain = load_fx(default_voices_dir(), voice).filter_chain()
        if engine not in CLONE_ENGINES:
            # готовый голос: темп - после синтеза (у движка нет параметра темпа;
            # до 3.7 книга голосом с сохранённым темпом падала - поймано стендом)
            from .stream import with_tempo
            fx_chain = with_tempo(fx_chain, load_fx(default_voices_dir(), voice).speed or 1.0)
    except Exception as e:
        log_problem(f"не прочитались настройки звучания голоса «{voice}», "
                    "книга будет без них", e)

    book_title = clean_book_title(doc.title or doc.base_title)
    narrator = opts.narrator or voice          # чтец в тегах (имя голоса или своё, «✎»)
    base = _safe_name(book_title)
    book_dir = os.path.join(out_dir, base)
    os.makedirs(book_dir, exist_ok=True)
    save_read_text(book_dir, prepared)
    save_disputed(book_dir, take_disputed())
    save_chunk_list(book_dir, prepared, opts.gap_sec, path)
    mp3_dir = book_dir if opts.make_chapter_mp3 else None
    # сырые части — в «.herald» рядом с результатом (см. «продолжение после сбоя»)
    workdir = os.path.join(book_dir, STATE_DIR)
    os.makedirs(workdir, exist_ok=True)
    fp = settings_fingerprint(engine, eng_kwargs, voice, opts)
    state = _load_state(workdir)
    if state.get("fingerprint") != fp or state.get("total") != len(prepared):
        state = {"fingerprint": fp, "total": len(prepared), "parts": {},
                 "finished": False}
    hashes = {ci: _part_hash(fp, chunks) for ci, (_t, chunks) in enumerate(prepared, 1)}
    rendered: dict[int, RenderedChapter] = {}
    workers = max(1, int(opts.workers))
    stats: dict = {}                    # свежий звук и куски из кэша (3.8)

    def emit(ci: int, r: RenderedChapter, expected_chars: int = 0,
             reuse: bool = False):
        """Потоковая выдача готового файла: пишем mp3 сразу и сообщаем наверх."""
        rendered[ci] = r
        if expected_chars and r.duration < expected_chars / 1000 * MIN_SEC_PER_1000:
            # НЕ перезаписываем прошлый удачный mp3 неудачным: если синтез
            # сорвался, лучше оставить что было и сказать об этом
            print(f"[!] файл {ci}: на {expected_chars} символов вышло всего "
                  f"{r.duration:.1f} с звука - синтез не сработал, файл не записан. "
                  f"Прежний результат (если был) остался на месте.", file=sys.stderr)
            if on_chapter:
                on_chapter(ci, r.title, None)
            return
        if mp3_dir:
            # имя файла — всегда «Часть N», у любой книги одинаково; настоящий
            # заголовок (если он есть) остаётся в тегах, плеер его покажет
            name = part_name(ci, len(prepared))
            ext = FORMATS.get(opts.audio_format, FORMATS[DEFAULT_FORMAT])[0]
            mp3 = os.path.join(mp3_dir, name + ext)
            tagged = RenderedChapter(
                name if not r.title or r.title.startswith("Часть ")
                else f"{name}. {r.title}", r.wav_path, r.duration)
            try:
                if not (reuse and os.path.exists(mp3)):
                    encode_chapter_mp3(tagged, mp3, title=book_title, author=doc.author,
                                       hd=opts.hd_filter, fmt=opts.audio_format,
                                       fx=fx_chain, narrator=narrator)
            except Exception as e:
                log_problem(f"не записался файл части {ci}", e)
                mp3 = None
            if on_chapter:
                on_chapter(ci, r.title, mp3)
        elif on_chapter:
            on_chapter(ci, r.title, None)
        if not reuse:
            state["parts"][str(ci)] = {"hash": hashes[ci],
                                       "flac": os.path.basename(r.wav_path),
                                       "dur": r.duration, "title": r.title}
            _save_state(workdir, state)

    def reusable(ci: int) -> RenderedChapter | None:
        """Готовая часть из прошлого запуска - если текст и настройки те же."""
        d = state["parts"].get(str(ci))
        if not d or d.get("hash") != hashes[ci]:
            return None
        fl = os.path.join(workdir, d.get("flac", ""))
        if not os.path.exists(fl):
            return None
        return RenderedChapter(d.get("title") or f"Часть {ci}", fl, float(d.get("dur", 0)))

    reused = {ci: r for ci in hashes if (r := reusable(ci)) is not None}
    if reused:
        log_info(f"{book_title}: продолжаю озвучку, готово частей {len(reused)} "
                 f"из {len(prepared)}")
        print(f"ℹ Продолжаю: готовых частей из прошлого раза - {len(reused)} "
              f"из {len(prepared)}", file=sys.stderr)

    # Mac не должен уснуть посреди книги: запрет держится до конца синтеза и
    # снимается при «Стопе» и при сбое (выход из блока — в любом случае)
    with keep_awake():
        if workers == 1:
            say(0.05, _model_msg(engine, eng_kwargs))
            try:
                vs = load_fx(default_voices_dir(), voice).speed
                if vs != 1.0 and engine in CLONE_ENGINES:
                    eng_kwargs = dict(eng_kwargs,
                                      speed=eng_kwargs.get("speed", 1.0) * vs)
            except Exception as e:
                log_problem(f"не прочитался темп голоса «{voice}», читаю обычным темпом", e)
            total_chunks = sum(max(1, len(c)) for _, c in prepared)
            done = sum(max(1, len(prepared[ci - 1][1])) for ci in reused)
            # модель грузим, только если есть что считать: продолжение, где всё
            # готово, не должно ждать загрузки 1,3 ГБ
            eng = _loaded_engine(engine, eng_kwargs) if len(reused) < len(prepared) else None
            if eng is not None and opts.sound_check and _clone(eng):
                _prepare_sound_check(say, eng_kwargs)
                _calibrate_stress(eng, voice, doc_lang, say, should_stop)
            paused = [0.0]

            def gate():
                """«Пауза» не должна попадать в замер скорости."""
                if pause_gate is not None:
                    t = time.monotonic()
                    pause_gate()
                    paused[0] += time.monotonic() - t
            for ci, (title, chunks) in enumerate(prepared, 1):
                if ci in reused:
                    emit(ci, reused[ci], reuse=True)
                    say(0.05 + 0.9 * done / total_chunks,
                        f"Часть {ci} из {len(prepared)} - готова с прошлого раза")
                    continue
                in_part = 0
                if part_progress:
                    part_progress(ci, len(prepared), 0.0)

                def bump():
                    nonlocal done, in_part
                    done += 1
                    in_part += 1
                    say(0.05 + 0.9 * done / total_chunks,
                        f"Озвучиваю: часть {ci} из {len(prepared)}")
                    if part_progress:
                        part_progress(ci, len(prepared), in_part / max(1, len(chunks)))
                t0, p0, f0 = time.monotonic(), paused[0], stats.get("fresh_sec", 0)
                from .stream import ENGINE_LOCK
                with ENGINE_LOCK:           # сервис /speak ждёт стыка частей
                    audio = _synth_chunks(eng, voice, chunks, sr, opts.gap_sec, on_chunk=bump,
                                          should_stop=should_stop, lang=doc_lang,
                                          pause_gate=gate, use_cache=opts.use_cache,
                                          strict_words=opts.strict_words,
                                          sound_check=opts.sound_check,
                                          stats=stats)
                spent = time.monotonic() - t0 - (paused[0] - p0)
                fresh = stats.get("fresh_sec", 0) - f0
                if fresh > 5:
                    rtf = spent / fresh
                    log_info(f"часть {ci}: RTF {rtf:.2f} (счёт {spent:.0f} с на "
                             f"{fresh:.0f} с нового звука; из кэша кусков: "
                             f"{stats.get('cached', 0)}, переозвучек: "
                             f"{stats.get('retries', 0)}, пауз доведено: "
                             f"{stats.get('pauses_fit', 0)})")
                    if on_rtf:
                        on_rtf(rtf)
                wp = os.path.join(workdir, f"part{ci:04d}.flac")
                emit(ci, RenderedChapter(title or f"Часть {ci}", wp, write_wav(audio, sr, wp)),
                     sum(len(c[0]) for c in chunks))
        else:
            import multiprocessing as mp
            try:
                import torch
                cores = torch.get_num_threads() or (os.cpu_count() or 4)
            except Exception:          # без torch просто берём число ядер — не важно
                cores = os.cpu_count() or 4
            threads_per = max(1, cores // workers)
            for ci in sorted(reused):
                emit(ci, reused[ci], reuse=True)
            payloads = [(ci, title, chunks, voice, sr, opts.gap_sec, opts.para_gap_sec, workdir)
                        for ci, (title, chunks) in enumerate(prepared, 1) if ci not in reused]
            say(0.05, f"Готовлю модель синтеза ({workers} процесса)…")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers, initializer=_worker_init,
                          initargs=(engine, eng_kwargs, threads_per)) as pool:
                done = 0
                # imap (а не imap_unordered): главы отдаются СТРОГО по порядку 1,2,3…,
                # даже если поздняя глава досчиталась раньше — чтобы слушать по порядку
                for ci, wav_path, dur, title in pool.imap(_worker_render, payloads):
                    emit(ci, RenderedChapter(title or f"Часть {ci}", wav_path, dur))
                    done += 1
                    say(0.05 + 0.9 * done / len(prepared), f"Озвучено глав: {done} из {len(prepared)}")

    ordered = [rendered[i] for i in sorted(rendered)]
    total_dur = sum(r.duration for r in ordered)

    single_path = None
    if opts.single_file and ordered:
        say(0.96, "Склеиваю части в один файл…")
        ext = FORMATS.get(opts.audio_format, FORMATS[DEFAULT_FORMAT])[0]
        folder = mp3_dir or out_dir
        single_path = os.path.join(folder, base + ext)
        concat_to_single(ordered, single_path, title=book_title, author=doc.author,
                         hd=opts.hd_filter, fmt=opts.audio_format, fx=fx_chain,
                         narrator=narrator)
        # по отдельности части больше не нужны: остаётся один файл
        if mp3_dir:
            for n in range(1, len(prepared) + 1):
                part = os.path.join(mp3_dir, part_name(n, len(prepared)) + ext)
                if os.path.exists(part) and part != single_path:
                    os.remove(part)

    m4b_path = None
    if opts.make_m4b:
        say(0.96, "Собираю аудиокнигу .m4b…")
        m4b_path = os.path.join(out_dir, base + ".m4b")
        assemble_m4b(ordered, m4b_path, title=book_title, author=doc.author, narrator=narrator,
                     cover_bytes=doc.cover.data if doc.cover else None,
                     cover_ext=doc.cover.ext if doc.cover else "jpg")

    # книга целиком готова: сырые части больше не нужны. До этой строки они
    # лежат в «.herald» — на случай сбоя, чтобы не считать книгу заново.
    if len(ordered) == len(prepared):
        shutil.rmtree(workdir, ignore_errors=True)
    say(1.0, "Готово")
    return ConvertResult(m4b_path, mp3_dir, 1 if single_path else len(ordered),
                         total_dur, book_title, doc.author,
                         float(stats.get("fresh_sec", 0.0)), int(stats.get("cached", 0)))


# --- оценка объёма книги (без синтеза) ----------------------------------
@dataclass
class BookEstimate:
    chapters: int
    chars: int
    audio_sec: float          # сколько будет звука
    title: str
    author: str
    language: str = "ru"      # от него зависит модель синтеза и ударения


def estimate_book(path: str, minutes_per_file: float = 15.0,
                  expand_numbers: bool = True) -> BookEstimate:
    """Сколько в книге глав и сколько выйдет звука - быстро, без синтеза.

    Нужно, чтобы интерфейс сразу честно сказал, на сколько часов это затянется:
    начитка книги - процесс на часы, и узнавать её длину по ходу неприятно.
    Ударения тут не считаем (RUAccent на всю книгу сам по себе не быстрый),
    а на длину они не влияют - только на произношение.
    """
    doc = read_document(path)
    chapters = build_chapters(doc, minutes_per_file)
    chars = sum(len(normalize_text(ch.text, expand_numbers=expand_numbers))
                for ch in chapters)
    return BookEstimate(len(chapters), chars, chars / CHARS_PER_SEC,
                        doc.title or doc.base_title, doc.author,
                        detect_language(doc.text or ""))


# --- пробник (~1 минута из середины) ------------------------------------
def preview_sample(path: str, *, engine: str = "f5", eng_kwargs: dict | None = None,
                   voice: str, minutes: float = 1.0, chapter_index: int | None = None,
                   out_path: str | None = None, options: ConvertOptions | None = None,
                   progress: Progress | None = None, live=None) -> str:
    """Пробник: минута из середины книги в wav. live (core/stream.Live) - звук
    ещё и потоком, по мере готовности; файл при этом тот же."""
    opts = options or ConvertOptions()

    def say(fr, msg):
        if progress:
            progress(max(0.0, min(1.0, fr)), msg)

    say(0.0, "Читаю файл…")
    doc = read_document(path)
    chapters = build_chapters(doc, opts.minutes_per_file)
    if not chapters:
        raise ReaderError("Не удалось извлечь текст из файла.")

    if chapter_index is not None:
        ch = chapters[max(0, min(chapter_index, len(chapters) - 1))]
    else:
        # берём самый длинный кусок: середина по счёту может оказаться огрызком,
        # и тогда пробник выходил пустым (0,2 с тишины)
        ch = max(chapters, key=lambda c: len(c.text))

    doc_lang = detect_language(doc.text or "")
    if engine in ("f5mlx", "f5", "f5torch") and "language" not in (eng_kwargs or {}):
        eng_kwargs = dict(eng_kwargs or {}, language=doc_lang)
    say(0.05, _model_msg(engine, eng_kwargs))
    # темп голоса подмешиваем в параметры, иначе кэш отдаст движок с прошлым
    try:
        from .engines.f5 import default_voices_dir
        vs = load_fx(default_voices_dir(), voice).speed
        if vs != 1.0 and engine in CLONE_ENGINES:
            eng_kwargs = dict(eng_kwargs or {},
                              speed=(eng_kwargs or {}).get("speed", 1.0) * vs)
    except Exception as e:
        log_problem(f"не прочитался темп голоса «{voice}», пробник обычным темпом", e)
    eng = _loaded_engine(engine, eng_kwargs)
    sr = eng.sample_rate
    if opts.sound_check and _clone(eng):
        _prepare_sound_check(say, {"language": doc_lang})

    text = _prepare_text(ch.text, getattr(eng, "stress_format", None),
                         opts.expand_numbers, opts.use_stress, doc_lang)
    # берём абзацы из середины целиком: так в пробнике слышны и паузы абзацев,
    # и паузы между репликами диалога - те же, что будут в книге
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    budget = int(minutes * 60 * CHARS_PER_SEC)
    picked, total, i = [], 0, len(paras) // 2
    while i < len(paras) and total < budget:
        para = paras[i]
        if total + len(para) > budget * 1.3:       # огромный абзац - по предложениям
            for sent in sentenize(para):
                if total >= budget:
                    break
                picked.append(sent); total += len(sent)
            break
        picked.append(para); total += len(para); i += 1
    if not picked and paras:
        picked = sentenize(paras[0])[:1]
    chunks = _chunk_paragraphs("\n\n".join(picked), eng.max_chunk_chars,
                               opts.gap_sec, opts.para_gap_sec, opts.dialog_gap_sec)
    if live is not None:
        from .stream import choose_first
        chunks = choose_first(chunks, live)  # первый звук - раньше, если это помогает

    say(0.2, "Озвучиваю пробный фрагмент…")
    done = [0]
    def bump():
        done[0] += 1
        say(0.2 + 0.7 * done[0] / max(1, len(chunks)), "Озвучиваю пробный фрагмент…")
    from .stream import ENGINE_LOCK
    try:
        with ENGINE_LOCK:
            audio = _synth_chunks(eng, voice, chunks, sr, opts.gap_sec, on_chunk=bump,
                                  lang=doc_lang, use_cache=opts.use_cache,
                                  strict_words=opts.strict_words,
                                  sound_check=opts.sound_check, live=live)
    except BaseException:
        if live is not None:
            live.stop()
        raise

    # временную папку пробника удаляет вызывающий (см. окно) или чистка при
    # запуске — она старше суток не живёт
    out_path = out_path or os.path.join(tempfile.mkdtemp(prefix="t2a_prev_"), "preview.wav")
    write_wav(audio, sr, out_path)
    say(1.0, "Пробник готов")
    return out_path
