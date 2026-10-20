"""meter2 - детектор ударения по звуку (siriusfreak, из репозитория модели
ru-stress-cf, папка meter/). 3.8, задание 10а, п. 3.

Звук + известный текст -> для каждого слова, какая гласная ударная. Точность на
многосложных словах по карточке: 98,3 % (LibriSpeech, новые чтецы), 99,3 %
(Silero, новые голоса) - точнее нашего детектора по образцам (core/stressdetect,
92 % на Silero).

Лицензия: у meter2 своей нет, обучен в том числе на звуке Silero (некоммерческая
лицензия) - автор просит считать его инструментом для исследований. Поэтому в
программу и в git он не входит: код (meter2.py) и «голова» (head.pt, 4 МБ)
скачиваются с Hugging Face при первом обращении в папку данных, как модели.
Выравниватель - jonatasgrosman/wav2vec2-large-xlsr-53-russian (Apache-2.0).
"""

from __future__ import annotations

import importlib.util
import os
import threading

REPO = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"
FILES = ("meter/meter2.py", "meter/head.pt")
_mod = None
_lock = threading.Lock()


def _dir() -> str:
    from .osutil import data_dir
    d = os.path.join(str(data_dir()), "meter2")
    os.makedirs(d, exist_ok=True)
    return d


def _load():
    global _mod
    if _mod is None:
        with _lock:
            if _mod is None:
                from huggingface_hub import hf_hub_download
                paths = {f: hf_hub_download(REPO, f) for f in FILES}
                os.environ.setdefault("METER2_HEAD", paths["meter/head.pt"])
                os.environ.setdefault("METER2_THREADS", "2")   # синтез на видеоядре не душим
                spec = importlib.util.spec_from_file_location("herald_meter2", paths["meter/meter2.py"])
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                _mod = m
    return _mod


def judge(audio, sr: int, text: str) -> list[dict]:
    """[{"word", "best", "margin", "vowel_scores", ...}] - как meter2.judge."""
    return _load().judge(audio, sr, text)


def marked_index(word_with_mark: str) -> int:
    """Номер ударной гласной в слове с «+» или U+0301 (0-based)."""
    from .stressdetect import vowel_index
    w = word_with_mark.replace("́", "")
    if "́" in word_with_mark:          # U+0301 стоит ПОСЛЕ гласной - перенести «+»
        i = word_with_mark.index("́")
        w = word_with_mark[:i - 1] + "+" + word_with_mark[i - 1] + word_with_mark[i + 1:]
    return vowel_index(w)


def wrong_words(audio, sr: int, text_marked: str, only: set[str] | None = None,
                min_margin: float = 0.3, timed: bool = False) -> list:
    """Слова, где ударение по звуку не там, где в тексте («+» или U+0301).
    only - проверять только эти слова (без ударения, нижний регистр);
    timed - [(слово, начало в записи, с)] вместо [слово] (11а, п. 5)."""
    import re
    marked = [w for w in re.findall(r"[А-Яа-яЁё+́]+(?:-[А-Яа-яЁё+́]+)*", text_marked)]
    res = judge(audio, sr, text_marked)
    out = []
    for r in res:
        k = r.get("word_index")
        if k is None or k >= len(marked) or r.get("best") is None:
            continue
        m = marked[k]
        if "+" not in m and "́" not in m:
            continue
        plain = m.replace("+", "").replace("́", "").lower()
        if only is not None and plain not in only:
            continue
        if r["best"] != marked_index(m) and (r.get("margin") or 0) >= min_margin:
            out.append((m, float(r.get("start") or 0.0)) if timed else m)
    return out
