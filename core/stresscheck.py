"""Проверка ударения по звуку (3.8, «упрямые слова»).

Текст до модели верный («+о́тмель»), а клон иногда читает по-своему. Приёмы
записи слова (знак ударения, удвоенная гласная, «а/и» вместо безударных, дефис,
заглавная) на замере 3.8 не помогли (bench_stubborn.py, Приёмка 3.8), зато
ошибка клона во многом случайна: то же слово с другим зерном читается верно.
Поэтому общий приём такой - как у проверки слов:

  1. слово «с риском» - редкое (нет в списке частых слов), омограф или уже
     замеченное программой как трудное для этой модели;
  2. после синтеза куска ударный слог такого слова определяется по звуку
     (core/stressdetect: сравнение с образцами Silero с ударением на каждом
     слоге);
  3. ударение не там - кусок пересчитывается с другим зерном (не больше
     STRESS_RETRIES раз), берётся дубль с верным ударением;
  4. слово, которое и после пересчётов звучит неверно, записывается в
     «трудные для модели» (папка данных, по ключу модели) - дальше оно
     проверяется всегда. Тот же список заполняется прогоном тестового набора
     при установке / обновлении модели (calibrate()).

Ручных словарей нет (решение Андрея 17.10.2026).
"""

from __future__ import annotations

import json
import os
import re
import threading

import numpy as np

from . import stressdetect as SD

CARRIER = "Скажи слово {} ещё раз."
REF_VOICES = ("aidar", "xenia")          # образцы - двумя голосами Silero
STRESS_RETRIES = 2                       # пересчётов ради ударения на кусок
NEW_REFS_PER_CHUNK = 4                   # новых образцов на кусок (процессор, ~1 с на слово)
MIN_MARGIN = 0.001                       # отрыв меньше - не уверены, пересчёт не просим

_common: set | None = None
_refs: dict[str, dict] = {}
_lock = threading.Lock()


def _assets() -> str:
    from .osutil import source_dir
    return os.path.join(str(source_dir()), "assets")


def common_words() -> set:
    global _common
    if _common is None:
        try:
            with open(os.path.join(_assets(), "частые_слова_ru.txt"), encoding="utf-8") as f:
                _common = {l.strip() for l in f if l.strip() and not l.startswith("#")}
        except OSError:
            _common = set()
    return _common


def _data_dir() -> str:
    from .osutil import data_dir
    return str(data_dir())


def hard_path() -> str:
    return os.path.join(_data_dir(), "трудные_ударения.json")


def hard_words(model_key: str) -> dict[str, int]:
    try:
        return json.load(open(hard_path(), encoding="utf-8")).get(model_key, {})
    except (OSError, ValueError):
        return {}


def remember_hard(model_key: str, words) -> None:
    """Слово звучало неверно и после пересчётов - запомнить для этой модели."""
    words = [w for w in words if w]
    if not words:
        return
    with _lock:
        try:
            data = json.load(open(hard_path(), encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        d = data.setdefault(model_key, {})
        for w in words:
            d[w] = d.get(w, 0) + 1
        tmp = hard_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, hard_path())


def _plain(w: str) -> str:
    return w.replace("+", "").lower().replace("ё", "е")


def is_homograph(plain: str) -> bool:
    try:
        from .stress import _get_accentizer
        acc = _get_accentizer()
        return acc is not None and plain in acc.omographs
    except Exception:
        return False


def risky(word_plus: str, hard: dict | None = None) -> int:
    """0 - не проверять; больше - важнее проверить (трудное > омограф > редкое)."""
    w = word_plus.lower()
    if "+" not in w or "ё" in w:
        return 0                                     # без ударения или «ё» - ясно
    plain = _plain(w)
    if sum(c in SD.VOWELS for c in plain) < 2:
        return 0
    if hard and plain in hard:
        return 3
    if is_homograph(plain):
        return 2
    return 1 if plain not in common_words() else 0


# --- образцы Silero ----------------------------------------------------------------
def _ref_file(plain: str) -> str:
    d = os.path.join(_data_dir(), "stress_refs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, plain + ".npz")


def refs_for(plain: str, aligner) -> dict[int, list[np.ndarray]]:
    """Образцы слова с ударением на каждом слоге (диск -> память -> Silero)."""
    if plain in _refs:
        return _refs[plain]
    p = _ref_file(plain)
    if os.path.exists(p):
        try:
            z = np.load(p)
            out: dict[int, list] = {}
            for k in z.files:
                out.setdefault(int(k.split("_")[0]), []).append(z[k].astype(np.float32))
            _refs[plain] = out
            return out
        except Exception:
            pass
    from .engines import get_engine
    from .align import _resample
    sil = get_engine("silero")
    sil.ensure_loaded()
    vpos = [i for i, c in enumerate(plain) if c in SD.VOWELS]
    out = {}
    save = {}
    for k, i in enumerate(vpos):
        for vi, voice in enumerate(REF_VOICES):
            au = sil.synth_chunk(CARRIER.format(plain[:i] + "+" + plain[i:]), voice)
            au = _resample(np.asarray(au, np.float32), sil.sample_rate, 24000)
            sg = segment_of(aligner, au, 24000, CARRIER.format(plain), plain)
            if sg is not None:
                out.setdefault(k, []).append(sg)
                save[f"{k}_{vi}"] = sg.astype(np.float16)
    if save:
        try:
            np.savez_compressed(_ref_file(plain), **save)
        except OSError:
            pass
    _refs[plain] = out
    return out


def segment_of(aligner, audio, sr, text, plain):
    ws = aligner.score(audio, sr, text)
    lp = aligner.last_lp
    for w in ws:
        if _plain(w.word) == plain and w.spans:
            return SD.segment(lp, w.spans, aligner._blank)
    return None


# --- проверка куска ----------------------------------------------------------------
def _parts(text: str) -> list[str]:
    """Слова текста с «+» - в том же порядке, что слова выравнивателя (дефис делит)."""
    out = []
    for m in re.finditer(r"[\w+\-]+", text):
        for part in m.group(0).split("-"):
            if part.replace("+", ""):
                out.append(part)
    return out


def wrong_words(ws, lp, text: str, aligner, hard: dict, budget: list) -> list[str]:
    """Слова куска, у которых ударение по звуку не там, где в тексте."""
    parts = _parts(text)
    if len(parts) != len(ws):
        return []                                    # выравнивание разошлось - не судим
    cand = [(risky(p, hard), j) for j, p in enumerate(parts)]
    cand = sorted((r, j) for r, j in cand if r)[::-1]
    wrong = []
    for r, j in cand:
        p, w = parts[j], ws[j]
        plain = _plain(p)
        if plain not in _refs and not os.path.exists(_ref_file(plain)):
            if budget[0] <= 0:
                continue                             # образцов на этот кусок хватит
            budget[0] -= 1
        if not w.spans:
            continue
        refs = refs_for(plain, aligner)
        if len(refs) < 2:
            continue
        pred, margin = SD.by_reference(SD.segment(lp, w.spans, aligner._blank), refs)
        if pred != SD.vowel_index(p) and margin >= MIN_MARGIN:
            wrong.append(p)
    return wrong


# --- тестовый набор модели -------------------------------------------------------
CARRIER_MARKED = "Скаж+и сл+ово {} ещё р+аз."


def test_set() -> list[str]:
    """Слова с «+» на верном месте: упрямые слова Андрея и редкие слова с ударением
    не на привычном месте (assets/ударения_тест.txt)."""
    try:
        with open(os.path.join(_assets(), "ударения_тест.txt"), encoding="utf-8") as f:
            return [l.split("|")[0].strip() for l in f if l.strip() and not l.startswith("#")]
    except OSError:
        return []


def calibrated(model_key: str) -> bool:
    try:
        return model_key in json.load(open(hard_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return False


def calibrate(eng, voice: str, aligner, progress=None, should_stop=None) -> list[str]:
    """Прогон тестового набора при установке / обновлении модели: клон читает
    каждое слово, ударение сверяется по звуку; неверные - в «трудные» для этой
    модели. Возвращает трудные слова."""
    key = eng.cache_key()
    words = test_set()
    bad = []
    for n, w in enumerate(words):
        if should_stop and should_stop():
            return bad
        if progress:
            progress(n / max(1, len(words)))
        plain = _plain(w)
        au = eng.synth_batch([CARRIER_MARKED.format(w)], voice)[0]
        sg = segment_of(aligner, au, eng.sample_rate, CARRIER.format(plain), plain)
        refs = refs_for(plain, aligner)
        if sg is None or len(refs) < 2:
            continue
        pred, margin = SD.by_reference(sg, refs)
        if pred != SD.vowel_index(w) and margin >= MIN_MARGIN:
            bad.append(plain)
    with _lock:
        try:
            data = json.load(open(hard_path(), encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        d = data.setdefault(key, {})
        for p in bad:
            d[p] = d.get(p, 0) + 1
        with open(hard_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    return bad
