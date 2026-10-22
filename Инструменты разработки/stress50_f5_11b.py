#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б (решение Андрея 22.10 10:24): ударения по звуку у F5 - прежний и дообученный.

Тот же набор, что stress50_39.py (Qwen): «ударения_50_39.txt», 50 трудных слов, фразы по 3 в
кусок; разметка - как в программе для F5 (`_prepare_text`, знак «+»), движок - профиль
«Основной» (f5mlx), одно зерно движка. Верно ли - meter2 против ВЕРНОГО ударения (ошибки
разметки текста тоже считаются). Голос со своими весами («.f5model») движок берёт сам.
Пишет «Архив/11б/ударения 50 F5/<голос>__NN.wav» и итог.json. Под ../.heavy-lock.

    python "Инструменты разработки/stress50_f5_11b.py" "Игорь Ященко" "Игорь Ященко (F5, обучен)" ...
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import re

OUT = "Архив/11б/ударения 50 F5"


def main() -> int:
    import soundfile as sf
    import core.pipeline as P
    from core import meter
    from core.engines import get_engine
    from core.profiles import PROFILES, engine_kwargs
    from core.stressdetect import vowel_index
    rows = [l.rstrip("\n").split("|") for l in open("Инструменты разработки/ударения_50_39.txt", encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    rows = [(p.strip(), w.strip(), t.strip()) for p, w, t in rows]
    chunks: list[list] = []                  # по 3 фразы, без повтора слова в куске (как stress50_39)
    for r in rows:
        for ch in chunks:
            if len(ch) < 3 and all(r[1].lower() != x[1].lower() for x in ch):
                ch.append(r)
                break
        else:
            chunks.append([r])
    marked = [P._with_end_punct(P._prepare_text(" ".join(p for p, _w, _t in ch), "+", True, True, "ru"))
              for ch in chunks]
    text_ok, text_bad = 0, []
    for ch, mt in zip(chunks, marked):
        for _p, w, t in ch:
            m = re.search(r"(?<![\w])\+?" + r"\+?".join(map(re.escape, w)) + r"(?![\w])", mt, re.IGNORECASE)
            good = bool(m) and vowel_index(m.group(0)) == vowel_index(t)
            text_ok += good
            if not good:
                text_bad.append(f"{w}: {m.group(0) if m else None} вместо {t}")
    print(f"разметка текста верна: {text_ok} из {len(rows)}", flush=True)
    p = PROFILES["main"]
    eng = get_engine(p.engine, language="ru", **engine_kwargs(p))
    eng.load()
    sr = eng.sample_rate
    _os.makedirs(OUT, exist_ok=True)
    res = {"разметка текста верна": f"{text_ok} из {len(rows)}", "разметка неверна": text_bad}
    for voice in _sys.argv[1:]:
        ok, n, wrong = 0, 0, []
        for ci, (ch, mt) in enumerate(zip(chunks, marked)):
            au = eng.synth_chunk(mt, voice)
            sf.write(_os.path.join(OUT, f"{voice}__{ci + 1:02d}.wav"), au, sr)
            truth = " ".join(re.sub(r"(?<![\w+])" + r"\+?".join(map(re.escape, w)) + r"(?![\w])", t, ph, count=1,
                                    flags=re.IGNORECASE) for ph, w, t in ch)
            words = {w.lower() for _p, w, _t in ch}
            bad = {b.replace("+", "").lower() for b in meter.wrong_words(au, sr, truth, only=words, min_margin=0.0)}
            ok += len(words - bad)
            n += len(words)
            wrong += sorted(bad)
        res[voice] = {"верно": ok, "из": n, "не так": wrong}
        print(f"{voice}: ударение верно {ok} из {n}; не так: {', '.join(wrong) or '-'}", flush=True)
    json.dump(res, open(_os.path.join(OUT, "итог.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
