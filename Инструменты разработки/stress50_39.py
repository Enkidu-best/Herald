#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, п. 1: ударения у Qwen ru-stress через нашу разметку - замер на 50 трудных словах.

Набор - «ударения_50_39.txt» (фраза | слово | верное ударение), фразы по 3 в кусок,
как режет программа. Голос - «Голос 01» папки 3.9 (образец - Архив/голоса диктора
3.9/образцы/Голос 06.wav), одно зерно на всё.
Условия:
  A  как сейчас: RUAccent (+ silero-stress у неуверенных омографов), знаки на всех словах
  B  без знаков на односложных словах («на», «он», «в»): два ударения подряд («на́ борту́»)
     модель сдвигает
  C  B + проверка meter2 после синтеза: ударение не там, где в разметке, - пересчёт
     куска, до двух раз
  D  C + без знака у спорных слов (RUAccent и silero-stress разошлись): такое слово
     модель читает сама - так ru-stress учили читать слова без знака
  E  B + словарь: слова, у которых разметка текста неверна, правятся по словарю
     (как если бы Андрей вписал их, увидев в «Как будет прочитано»); только замер -
     в программе ручного словаря нет (решение Андрея 17.10)
Верно ли ударение - по meter2 против ВЕРНОГО ударения (не против нашей разметки),
поэтому ошибки текста тоже считаются. Отдельно - верна ли сама разметка текста.
Под ../.heavy-lock.
    python "Инструменты разработки/stress50_39.py" [A B C]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import re

OUT = "Архив/голоса диктора 3.9/ударения 50"
VOICE = _os.path.abspath("Архив/голоса диктора 3.9/образцы/Голос 06.wav")
VOW = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def strip_short(text: str) -> str:
    """Знак ударения у односложных слов не нужен: «н+а борт+у» -> «на борт+у»."""
    def sub(m):
        w = m.group(0)
        return w.replace("+", "") if sum(c in VOW for c in w) <= 1 else w
    return re.sub(r"[А-Яа-яЁё+]+", sub, text)


def unmark(text: str, words: set) -> str:
    """Снять знак с этих слов (без ударения, нижний регистр, ё=е)."""
    def sub(m):
        w = m.group(0)
        return w.replace("+", "") if w.replace("+", "").lower().replace("ё", "е") in words else w
    return re.sub(r"[А-Яа-яЁё+]+", sub, text)


def main() -> int:
    import soundfile as sf
    import core.pipeline as P
    from core import meter
    from core.stressdetect import vowel_index
    from core.engines.mlxaudio import MlxAudioEngine
    rows = [l.rstrip("\n").split("|") for l in open("Инструменты разработки/ударения_50_39.txt", encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    rows = [(p.strip(), w.strip(), t.strip()) for p, w, t in rows]
    chunks: list[list] = []                  # по 3 фразы, без повтора слова в куске
    for r in rows:
        for ch in chunks:
            if len(ch) < 3 and all(r[1].lower() != x[1].lower() for x in ch):
                ch.append(r)
                break
        else:
            chunks.append([r])
    conds = _sys.argv[1:] or ["A", "B", "C"]
    eng = MlxAudioEngine(model="qwen-stress", bits=8, seed=7)
    eng.ensure_loaded()
    res = {}
    _os.makedirs(OUT, exist_ok=True)
    # разметка текста (одна для всех условий)
    from core.stress import take_disputed
    marked, disputed = [], []
    for ch in chunks:
        take_disputed()
        marked.append(P._with_end_punct(P._prepare_text(" ".join(p for p, _w, _t in ch), "+", True, True, "ru")))
        disputed.append(set(take_disputed()))
    text_ok, wrong = 0, []
    for ci, (ch, mt) in enumerate(zip(chunks, marked)):
        for _p, w, t in ch:
            m = re.search(r"(?<![\w])\+?" + r"\+?".join(map(re.escape, w)) + r"(?![\w])", mt, re.IGNORECASE)
            good = bool(m) and vowel_index(m.group(0)) == vowel_index(t)
            text_ok += good
            if not good:
                wrong.append((ci, w, t, m.group(0) if m else None))
    res["разметка текста верна"] = f"{text_ok} из {len(rows)}"
    res["разметка неверна"] = [f"{w}: {got} вместо {t}" for _ci, w, t, got in wrong]
    print("разметка текста верна:", res["разметка текста верна"], flush=True)
    for cond in conds:
        ok, n, retries = 0, 0, 0
        for ci, (ch, mt) in enumerate(zip(chunks, marked)):
            text = strip_short(mt) if cond in ("B", "C", "D") else mt
            if cond == "D":
                text = unmark(text, disputed[ci])
            if cond == "E":
                for wci, w, t, got in wrong:
                    if wci == ci:
                        text = re.sub(r"(?<![\w+])" + r"\+?".join(map(re.escape, w)) + r"(?![\w])",
                                      t, text, count=1, flags=re.IGNORECASE)
            au = eng.synth_chunk(text, VOICE)
            if cond in ("C", "D"):
                for _att in range(2):
                    bad = meter.wrong_words(au, eng.sample_rate, text)
                    if not bad:
                        break
                    retries += 1
                    au = eng.synth_chunk(text, VOICE)
            sf.write(_os.path.join(OUT, f"{cond}__{ci + 1:02d}.wav"), au, eng.sample_rate)
            # верно ли по звуку - против верного ударения
            truth = " ".join(re.sub(r"(?<![\w+])" + r"\+?".join(map(re.escape, w)) + r"(?![\w])", t, p, count=1,
                                    flags=re.IGNORECASE) for p, w, t in ch)
            words = {w.lower() for _p, w, _t in ch}
            bad = {b.replace("+", "").lower() for b in meter.wrong_words(au, eng.sample_rate, truth, only=words,
                                                                          min_margin=0.0)}
            ok += len(words - bad)
            n += len(words)
        res[cond] = {"верно": ok, "из": n, "пересчётов": retries}
        print(f"{cond}: ударение верно {ok} из {n}, пересчётов {retries}", flush=True)
    eng.close()
    path = _os.path.join(OUT, "итог.json")
    old = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    old.update(res)
    res = old
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
