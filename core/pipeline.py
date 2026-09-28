"""Оркестровка: файл книги -> аудиокнига, плюс быстрый «пробник».

Синтез глав можно вести параллельно на нескольких ядрах (workers>1) - каждый
рабочий процесс поднимает свой движок и озвучивает свою пачку глав. Подготовку
текста (нормализация, ударения) делаем один раз в родителе.

GUI и CLI вызывают convert_book() (вся книга) или preview_sample() (~1 минута).
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .readers import read_document, ScannedPdfError, ReaderError
from .chapters import build_chapters, CHARS_PER_SEC
from .normalize import normalize_text, sentenize, detect_language
from .stress import add_stress
from .assemble import (RenderedChapter, write_wav, assemble_m4b, encode_chapter_mp3,
                       polish_wav_to_mp3, FORMATS, DEFAULT_FORMAT)
from .engines import get_engine, TTSEngine
from .engines.f5 import default_voices_dir
from .voicefx import load as load_fx


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
        _ENGINE_CACHE.clear()          # держим не больше одной модели разом
        _ENGINE_CACHE[key] = eng
    return eng


@dataclass
class ConvertOptions:
    make_m4b: bool = False           # единый файл-книга — по желанию
    make_chapter_mp3: bool = True    # по умолчанию отдельные mp3 по главам (потоково)
    minutes_per_file: float = 15.0   # примерная длина одного mp3 по звуку
    hd_filter: bool = True           # полировка звука перед кодированием (см. assemble)
    audio_format: str = DEFAULT_FORMAT
    expand_numbers: bool = True
    use_stress: bool = True
    gap_sec: float = 0.35
    para_gap_sec: float = 0.6
    workers: int = 1                 # параллельные процессы для синтеза глав


@dataclass
class ConvertResult:
    m4b_path: Optional[str]
    mp3_dir: Optional[str]
    chapters: int
    duration: float
    title: str
    author: str


def _safe_name(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", " ", s).strip()
    return re.sub(r"\s{2,}", " ", s) or "audiobook"


def _prepare_text(text: str, stress_fmt: str | None, expand_numbers: bool,
                  use_stress: bool, language: str = "ru") -> str:
    t = normalize_text(text, expand_numbers=expand_numbers, language=language)
    # ударения ставим только русскому: английская модель прочитала бы «+» вслух
    if use_stress and stress_fmt == "+" and language == "ru":
        t = add_stress(t)
    return t


def _chunk_sentences(text: str, max_chars: int) -> list[str]:
    chunks, cur = [], ""
    for sent in sentenize(text):
        if len(sent) > max_chars:
            for piece in _hard_split(sent, max_chars):
                if cur:
                    chunks.append(cur); cur = ""
                chunks.append(piece)
            continue
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
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


def _is_heading(p: str) -> bool:
    """Короткая строка без точки на конце — заголовок (та же мерка, что в chapters)."""
    return (len(p) <= 90 and p[-1] not in ".!?…:,;»\"'"
            and not p[0].islower())


def _chunk_paragraphs(text: str, max_chars: int, gap: float,
                      para_gap: float) -> list[tuple[str, float, float]]:
    """[(кусок текста, пауза после него)]. Между абзацами пауза длиннее.

    Раньше вся глава резалась на куски одной лентой и между ними всегда стояла
    одна и та же пауза 0,35 с. На слух это слипалось: заголовок переходил в
    первый абзац без остановки, а смена абзаца ничем не отличалась от середины
    предложения. Теперь конец абзаца звучит как конец абзаца.
    """
    out: list[tuple[str, float, float]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        # заголовок не склеиваем с текстом и читаем его размереннее: иначе он
        # проскакивает скороговоркой, а последнее слово в нём пропадает
        if _is_heading(para) and len(para) >= MIN_HEADING_CHARS:
            out.append((para, para_gap, HEADING_SPEED_MUL))
            continue
        parts = _chunk_sentences(para, max_chars)
        for i, c in enumerate(parts):
            out.append((c, gap if i < len(parts) - 1 else para_gap, 1.0))
    return _merge_short(out, max_chars)


# Куски меньше этой доли от максимума склеиваем со следующими. Причина в
# арифметике: на каждый вызов модель заново прогоняет весь образец (около 10 с),
# поэтому кусок на 95 символов стоит почти столько же, сколько на 250, а звука
# даёт втрое меньше. Замерено на книге: куски по 95 символов — синтез 162 с,
# по 200+ — около 110 с при том же объёме звука.
MERGE_UNDER_FRACTION = 0.95
MIN_CHUNK_CHARS = 45
# Заголовок короче этого отдельным куском не оставляем: на коротком отрезке
# модели не за что зацепиться, и «Проба голоса» превращается в «Проба ГОД», а
# «Глава первая. Ветер с залива» пропадает целиком. Склеенный с первым абзацем
# заголовок читается надёжно, паузу после него даёт точка.
MIN_HEADING_CHARS = 45


def _merge_short(items, max_chars: int):
    """Склеить куски короче MIN_CHUNK_CHARS со следующим.

    На огрызках вроде «Глава 5.» или «Открывая пустыню» F5 ведёт себя плохо:
    материала мало, и модель то глотает слово целиком, то мажет окончание.
    Пауза между склеенными кусками сохраняется — просто она теперь внутри
    одного вызова, и модель произносит их как одну фразу с остановкой.
    """
    out: list[tuple[str, float, float]] = []
    limit = int(max_chars * MERGE_UNDER_FRACTION)
    for text, gap, mul in items:
        prev_is_heading = out and out[-1][2] != 1.0
        if (out and not prev_is_heading and mul == 1.0
                and len(out[-1][0]) < limit
                and len(out[-1][0]) + len(text) + 2 <= max_chars):
            prev, _g, pm = out[-1]
            sep = " " if prev.endswith((".", "!", "?", "…", ",", ":", ";")) else ". "
            out[-1] = (prev + sep + text, gap, pm)
        else:
            out.append((text, gap, mul))
    return out


# --- страховка от съеденных краёв ----------------------------------------
# F5 иногда мажет у самых границ куска: то последнее слово теряет окончание
# («ветер» -> «веть»), то пропадает первое («Море к вечеру стихло» -> «стихло»).
# Каждый синтез стартует со своего случайного шума, поэтому достаточно
# переспросить — со второй попытки кусок обычно выходит чистым. Проверяем
# ТОЛЬКО крайние слова: распознавание всего куска стоило бы дороже синтеза.
TAIL_RETRIES = 2
# Модель для проверки краёв. base (130 мс) оказалась слишком слабой: она врала
# на краях («глова» вместо «глава», «гакма» вместо «как море»), и половина
# кусков пересинтезировалась впустую. small — 310 мс и уже не путается, против
# 1260 мс у large-v3-turbo. Плюс сравниваем слова НЕЧЁТКО: распознаватель имеет
# право слегка переврать окончание, нам важно, что слово вообще прозвучало.
_ASR_MODEL = "mlx-community/whisper-small-mlx"
_WORD_SIMILAR = 0.75
_asr_failed = False


def _last_word(s: str) -> str:
    words = re.sub(r"[^\w ]", " ", s.lower().replace("+", "").replace("ё", "е")).split()
    return words[-1] if words else ""


def _words(s: str) -> list[str]:
    return re.sub(r"[^\w ]", " ", s.lower().replace("+", "").replace("ё", "е")).split()


def _sounds_like(want: str, heard: list[str]) -> bool:
    """Есть ли среди услышанных слов похожее на нужное.

    Сравниваем нечётко и ещё прощаем потерю первой буквы: модель смазывает
    самую атаку звука на стыке кусков, и «проба» слышится как «роба». Гнать
    из-за одной буквы кусок на пересинтез бессмысленно — второй раз выйдет
    то же самое, а время потратим.
    """
    import difflib
    for w in heard:
        if difflib.SequenceMatcher(None, want, w).ratio() >= _WORD_SIMILAR:
            return True
        if len(want) > 3 and (w == want[1:] or w == want[:-1]):
            return True
    return False


def _edges_ok(audio: np.ndarray, sr: int, text: str, head_only: bool = False,
              lang: str = "ru") -> bool:
    """Слышны ли первое и последнее слова куска — те самые, что в тексте."""
    global _asr_failed
    if _asr_failed or len(audio) < sr:
        return True
    want = _words(text)
    if not want:
        return True
    try:
        import contextlib
        import io
        import tempfile as _tf
        import soundfile as _sf
        import mlx_whisper

        def heard(chunk):
            with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                _sf.write(f.name, chunk, sr)
                # язык передаём явно: иначе whisper определяет его сам, и это
                # ровно удваивает время проверки (2,4 с против 1,3 с)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = mlx_whisper.transcribe(f.name, path_or_hf_repo=_ASR_MODEL,
                                               language=lang, verbose=False)
            os.unlink(f.name)
            return _words(r["text"])

        # слова короче четырёх букв распознаватель путает сам — их не проверяем
        if len(want[0]) >= 4:
            # Начало слушаем коротко (2 с) и ВСЕГДА: пропажу первого слова по
            # громкости не поймать — звук на месте, просто произносится уже
            # следующее слово. Двух секунд хватает, и это дёшево.
            got = heard(audio[:int(min(len(audio) / sr, 2.0) * sr)])
            if got and not _sounds_like(want[0], got[:2]):
                return False
        if not head_only and len(want[-1]) >= 4:
            edge = int(min(len(audio) / sr, 4.0) * sr)
            got = heard(audio[-edge:])
            if got and not _sounds_like(want[-1], got[-2:]):
                return False
        return True
    except Exception:
        _asr_failed = True                  # нет модели — просто не проверяем
        return True


def _synth_chunks(eng: TTSEngine, voice: str, chunks, sr: int,
                  gap_sec: float, on_chunk=None, should_stop=None,
                  lang: str = "ru") -> np.ndarray:
    # принимаем простые строки, пары (текст, пауза) и тройки (…, множитель темпа)
    items = []
    for c in chunks:
        if isinstance(c, str):
            items.append((c, gap_sec, 1.0))
        elif len(c) == 2:
            items.append((c[0], c[1], 1.0))
        else:
            items.append(tuple(c))
    pieces: list[np.ndarray] = []
    for text, gap, mul in items:
        # проверяем ПЕРЕД куском: один кусок считается секунды, ждать не придётся
        if should_stop is not None and should_stop():
            raise Cancelled()
        audio = eng.synth_chunk(text, voice, speed_mul=mul)
        for _try in range(TAIL_RETRIES):
            if not len(audio):
                break
            # Проверяем ОБА края и всегда. Раньше конец смотрели только при
            # подозрении по громкости, но смазанное окончание звучит не тише —
            # «восемь часов» превращалось в «восемь чеф», и фильтр это
            # пропускал. С моделью small проверка стоит около 0,3 с на край,
            # то есть считаные проценты синтеза — дешевле, чем брак.
            if _edges_ok(audio, sr, text, lang=lang):
                break
            audio = eng.synth_chunk(text, voice, speed_mul=mul)
        if len(audio):
            pieces.append(audio)
            pieces.append(np.zeros(int(sr * gap), dtype=np.float32))
        if on_chunk:
            on_chunk()
    return np.concatenate(pieces) if pieces else np.zeros(int(sr * 0.2), dtype=np.float32)


# --- параллельные рабочие процессы --------------------------------------
_W: dict = {}


def _worker_init(engine_id: str, eng_kwargs: dict, threads: int) -> None:
    try:
        import torch
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass
    eng = get_engine(engine_id, **(eng_kwargs or {}))
    eng.ensure_loaded()
    _W["eng"] = eng


def _worker_render(payload: tuple) -> tuple:
    ci, title, chunks, voice, sr, gap_sec, para_gap, workdir = payload
    eng = _W["eng"]
    audio = _synth_chunks(eng, voice, chunks, sr, gap_sec)
    tail = np.zeros(int(sr * para_gap), dtype=np.float32)
    audio = np.concatenate([audio, tail])
    wav_path = os.path.join(workdir, f"ch{ci:04d}.wav")
    dur = write_wav(audio, sr, wav_path)
    return ci, wav_path, dur, title


# --- вся книга -----------------------------------------------------------
def convert_book(path: str, *, engine: str = "f5", eng_kwargs: dict | None = None,
                 voice: str, out_dir: str | None = None,
                 options: ConvertOptions | None = None,
                 progress: Progress | None = None,
                 on_chapter: Callable | None = None,
                 should_stop: Callable[[], bool] | None = None) -> ConvertResult:
    """Озвучить книгу. Главы выдаются потоково: как только глава готова, её mp3
    сразу пишется рядом и вызывается on_chapter(index, title, mp3_path) - можно
    начинать слушать, пока остальные считаются. В конце собирается единый .m4b.
    """
    opts = options or ConvertOptions()
    eng_kwargs = eng_kwargs or {}
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    # язык книги выбирает и модель синтеза, и нужны ли ударения
    doc_lang = detect_language(read_document(path).text or "")
    if engine in ("f5mlx", "f5") and "language" not in eng_kwargs:
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
    prepared = []
    for ch in chapters:
        # заголовок уже стоит первой строкой в ch.text (см. chapters.py), отдельно
        # его добавлять нельзя — иначе название прочитается дважды
        body = _prepare_text(ch.text, stress_fmt, opts.expand_numbers,
                             opts.use_stress, doc_lang)
        prepared.append((ch.title, _chunk_paragraphs(body, meta.max_chunk_chars,
                                                     opts.gap_sec, opts.para_gap_sec)))

    # настройки голоса: тембр применяем при кодировании, темп — при синтезе
    fx_chain = ""
    try:
        fx_chain = load_fx(default_voices_dir(), voice).filter_chain()
    except Exception:
        pass

    base = _safe_name(doc.base_title)
    workdir = tempfile.mkdtemp(prefix="t2a_syn_")
    # папку под потоковые mp3 создаём сразу, если они нужны
    mp3_dir = None
    if opts.make_chapter_mp3:
        mp3_dir = os.path.join(out_dir, base)
        os.makedirs(mp3_dir, exist_ok=True)
    rendered: dict[int, RenderedChapter] = {}
    workers = max(1, int(opts.workers))

    def emit(ci: int, r: RenderedChapter, expected_chars: int = 0):
        """Потоковая выдача готового файла: пишем mp3 сразу и сообщаем наверх."""
        rendered[ci] = r
        if expected_chars and r.duration < expected_chars / 1000 * MIN_SEC_PER_1000:
            # НЕ перезаписываем прошлый удачный mp3 неудачным: если синтез
            # сорвался, лучше оставить что было и сказать об этом
            print(f"[!] файл {ci}: на {expected_chars} символов вышло всего "
                  f"{r.duration:.1f} с звука — синтез не сработал, файл не записан. "
                  f"Прежний результат (если был) остался на месте.", file=sys.stderr)
            if on_chapter:
                on_chapter(ci, r.title, None)
            return
        if mp3_dir:
            name = _safe_name(f"{ci:02d} {r.title}")[:80]
            ext = FORMATS.get(opts.audio_format, FORMATS[DEFAULT_FORMAT])[0]
            mp3 = os.path.join(mp3_dir, name + ext)
            try:
                encode_chapter_mp3(r, mp3, title=doc.title or base, author=doc.author,
                                   hd=opts.hd_filter, fmt=opts.audio_format,
                                   fx=fx_chain)
            except Exception:
                mp3 = None
            if on_chapter:
                on_chapter(ci, r.title, mp3)
        elif on_chapter:
            on_chapter(ci, r.title, None)

    if workers == 1:
        say(0.05, "Готовлю модель синтеза…")
        try:
            vs = load_fx(default_voices_dir(), voice).speed
            if vs != 1.0:
                eng_kwargs = dict(eng_kwargs,
                                  speed=eng_kwargs.get("speed", 1.0) * vs)
        except Exception:
            pass
        eng = _loaded_engine(engine, eng_kwargs)
        total_chunks = sum(max(1, len(c)) for _, c in prepared)
        done = 0
        for ci, (title, chunks) in enumerate(prepared, 1):
            def bump():
                nonlocal done
                done += 1
                say(0.05 + 0.9 * done / total_chunks, f"Озвучиваю: глава {ci} из {len(prepared)}")
            audio = _synth_chunks(eng, voice, chunks, sr, opts.gap_sec, on_chunk=bump,
                                  should_stop=should_stop, lang=doc_lang)
            audio = np.concatenate([audio, np.zeros(int(sr * opts.para_gap_sec), dtype=np.float32)])
            wp = os.path.join(workdir, f"ch{ci:04d}.wav")
            emit(ci, RenderedChapter(title or f"Часть {ci}", wp, write_wav(audio, sr, wp)),
                 sum(len(c[0]) for c in chunks))
    else:
        import multiprocessing as mp
        try:
            import torch
            cores = torch.get_num_threads() or (os.cpu_count() or 4)
        except Exception:
            cores = os.cpu_count() or 4
        threads_per = max(1, cores // workers)
        payloads = [(ci, title, chunks, voice, sr, opts.gap_sec, opts.para_gap_sec, workdir)
                    for ci, (title, chunks) in enumerate(prepared, 1)]
        say(0.05, f"Готовлю модель синтеза ({workers} процесса)…")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_worker_init,
                      initargs=(engine, eng_kwargs, threads_per)) as pool:
            done = 0
            # imap (а не imap_unordered): главы отдаются СТРОГО по порядку 1,2,3…,
            # даже если поздняя глава досчиталась раньше — чтобы слушать по порядку
            for ci, wav_path, dur, title in pool.imap(_worker_render, payloads):
                emit(ci, RenderedChapter(title or f"Глава {ci}", wav_path, dur))
                done += 1
                say(0.05 + 0.9 * done / len(prepared), f"Озвучено глав: {done} из {len(prepared)}")

    ordered = [rendered[i] for i in sorted(rendered)]
    total_dur = sum(r.duration for r in ordered)

    m4b_path = None
    if opts.make_m4b:
        say(0.96, "Собираю аудиокнигу .m4b…")
        m4b_path = os.path.join(out_dir, base + ".m4b")
        assemble_m4b(ordered, m4b_path, title=doc.title or base, author=doc.author,
                     cover_bytes=doc.cover.data if doc.cover else None,
                     cover_ext=doc.cover.ext if doc.cover else "jpg")

    say(1.0, "Готово")
    return ConvertResult(m4b_path, mp3_dir, len(ordered), total_dur, doc.title or base, doc.author)


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
    """Сколько в книге глав и сколько выйдет звука — быстро, без синтеза.

    Нужно, чтобы интерфейс сразу честно сказал, на сколько часов это затянется:
    начитка книги — процесс на часы, и узнавать её длину по ходу неприятно.
    Ударения тут не считаем (RUAccent на всю книгу сам по себе не быстрый),
    а на длину они не влияют — только на произношение.
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
                   progress: Progress | None = None) -> str:
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
    if engine in ("f5mlx", "f5") and "language" not in (eng_kwargs or {}):
        eng_kwargs = dict(eng_kwargs or {}, language=doc_lang)
    say(0.05, "Готовлю модель синтеза…")
    # темп голоса подмешиваем в параметры, иначе кэш отдаст движок с прошлым
    try:
        from .engines.f5 import default_voices_dir
        vs = load_fx(default_voices_dir(), voice).speed
        if vs != 1.0:
            eng_kwargs = dict(eng_kwargs or {},
                              speed=(eng_kwargs or {}).get("speed", 1.0) * vs)
    except Exception:
        pass
    eng = _loaded_engine(engine, eng_kwargs)
    sr = eng.sample_rate

    text = _prepare_text(ch.text, getattr(eng, "stress_format", None),
                         opts.expand_numbers, opts.use_stress, doc_lang)
    sents = sentenize(text)
    budget = int(minutes * 60 * CHARS_PER_SEC)
    mid = len(sents) // 2
    picked, total, i = [], 0, mid
    while i < len(sents) and total < budget:
        picked.append(sents[i]); total += len(sents[i]); i += 1
    if not picked:
        picked = sents[:1]
    chunks = _chunk_paragraphs(" ".join(picked), eng.max_chunk_chars,
                               opts.gap_sec, opts.para_gap_sec)

    say(0.2, "Озвучиваю пробный фрагмент…")
    done = [0]
    def bump():
        done[0] += 1
        say(0.2 + 0.7 * done[0] / max(1, len(chunks)), "Озвучиваю пробный фрагмент…")
    audio = _synth_chunks(eng, voice, chunks, sr, opts.gap_sec, on_chunk=bump,
                          lang=doc_lang)

    out_path = out_path or os.path.join(tempfile.mkdtemp(prefix="t2a_prev_"), "preview.wav")
    write_wav(audio, sr, out_path)
    say(1.0, "Пробник готов")
    return out_path
