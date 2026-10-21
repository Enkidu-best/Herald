#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 1: темп предложения в зависимости от его места ВНУТРИ КУСКА синтеза.

Гипотеза «желе - пулемёт» (оценка Андрея 21.10): модель начинает кусок медленно и к его
середине разгоняется. Кусок у Qwen - до 600 знаков (несколько предложений), у F5 - 187.
По длинному куску («Глава первая») - слова Whisper с метками времени, сверенные с текстом;
для каждого предложения - слогов в секунду (от первого до последнего слова) и номер
предложения в его куске (куски - как режет конвейер для этого движка).

    python "Инструменты разработки/tempo_in_chunk_11b.py" <mp3> <знаков в куске> [подпись]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import importlib.util
import json
import re

import numpy as np

SRC = "Архив/голоса диктора 3.9/длинный кусок/Глава первая (отрывок).txt"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    import core.pipeline as P
    from razdel import sentenize
    M = _mod("m11b", "Инструменты разработки/measure_11b.py")
    sim = _mod("sim38", "Инструменты разработки/bench_similarity_38.py")
    path, maxc = _sys.argv[1], int(_sys.argv[2])
    label = _sys.argv[3] if len(_sys.argv) > 3 else _os.path.basename(path)
    text = open(SRC, encoding="utf-8").read()
    a16 = sim.load16(path)
    hw = M.heard(P, a16)
    sents, per, _m, _n = M.sentences_with_words(text, hw)
    # номер предложения в куске: куски конвейера по тексту без разметки
    chunks = P._chunk_paragraphs(text, maxc, 0.4, 0.8, 1.1)
    pos, si = {}, 0
    for c in chunks:
        cs = [s.text for s in sentenize(c[0])]
        for k, _s in enumerate(cs):
            pos[si] = k
            si += 1
    by = {}
    for i, ws in per.items():
        if len(ws) < 4 or i not in pos:
            continue
        dur = ws[-1][2] - ws[0][1]
        if dur < 1.0:
            continue
        rate = sum(M.syl(w) for w, _s, _e in ws) / dur
        by.setdefault(min(pos[i], 3), []).append(rate)
    res = {"вариант": label, "куски до": maxc,
           "слогов/с по месту в куске (1-е, 2-е, 3-е, 4-е и дальше)":
               {str(k + 1): [round(float(np.median(v)), 2), len(v)] for k, v in sorted(by.items())}}
    if 0 in by and len(by) > 1:
        later = [x for k, v in by.items() if k > 0 for x in v]
        res["дальше / первое"] = round(float(np.median(later) / np.median(by[0])), 2)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    with open("Архив/11б/замеры.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
