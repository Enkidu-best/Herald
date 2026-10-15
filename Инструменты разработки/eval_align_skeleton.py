#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Порог по «согласному скелету» услышанного (3.3) - на сохранённых оценках.

Слово под подозрением, если (а) вероятность его букв очень низкая (post < P1) -
слово пропало или съедено, или (б) акустическая модель без языковой догадки
слышит на его месте ДРУГИЕ согласные (сходство скелетов < S) при заметно
сниженной вероятности (post < P2). Гласные и мягкий/твёрдый знак, й - не
сравниваются: безударные гласные русская речь редуцирует («абориген» ->
«обреген»), и это не ошибка.

    python "Инструменты разработки/eval_align_skeleton.py" слова.json
"""
from __future__ import annotations

import difflib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from eval_align import apply_labels                     # noqa: E402
from core.align import skeleton_sim                    # noqa: E402


def run(rows, P1, P2, S, gap_thr, ml):
    flagged = set()
    for r in rows:
        if r["букв"] < ml:
            continue
        sk = skeleton_sim(r["слово"], r.get("слышно", ""))
        if r["post"] < P1 or (r["post"] < P2 and sk < S):
            flagged.add((r["файл"], r["i"]))
        if gap_thr > 0 and r["знак"] in ",.!?…:;—–" and r["gap"] < gap_thr:
            flagged.add((r["файл"], r["i"]))
    caught, labeled = [], set()
    for r in rows:
        for lab in r.get("метки", []):
            labeled.add(r["кусок"])
            f, i = r["файл"], r["i"]
            if lab["вид"] == "слитно":
                ok = gap_thr > 0 and r["gap"] < gap_thr
            else:
                ok = any((f, j) in flagged for j in (i - 1, i, i + 1))
            caught.append((lab["id"], f, ok))
    chunks = {r["кусок"] for r in rows}
    bad = {r["кусок"] for r in rows if (r["файл"], r["i"]) in flagged}
    fp_chunks = sorted(bad - labeled)
    return (sum(o for *_x, o in caught) / max(1, len(caught)),
            len(fp_chunks) / max(1, len(chunks - labeled)), caught, fp_chunks, flagged)


def main():
    rows = json.load(open(sys.argv[1], encoding="utf-8"))
    apply_labels(rows, json.load(open(os.path.join(HERE, "разметка_3.3", "набор.json"), encoding="utf-8")))
    comma = sorted(r["gap"] for r in rows if r["знак"] == "," and not r.get("метки"))
    if comma:
        print(f"пауза после запятой у чистых слов: 5-й процентиль {comma[len(comma)//20]:.2f}, медиана {comma[len(comma)//2]:.2f}")
    res = []
    for P1 in (-3.0, -3.5, -4.0, -5.0, -6.0):
        for P2 in (-0.5, -0.8, -1.0, -1.5, -2.0):
            for S in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85):
                for g in (0.0, 0.1, 0.13):
                    for ml in (1, 2):
                        rec, fp, caught, fpc, _f = run(rows, P1, P2, S, g, ml)
                        res.append((rec, -fp, P1, P2, S, g, ml, fp, caught, fpc))
    # прежняя строгая проверка 3.2 на том же наборе
    old = {(r["файл"], r["i"]) for r in rows if r.get("старая")}
    if any(r.get("старая") is not None for r in rows):
        cau, lab_chunks = [], set()
        for r in rows:
            for lab in r.get("метки", []):
                lab_chunks.add(r["кусок"])
                cau.append((lab["id"], r["файл"], lab["вид"] != "слитно" and any(
                    (r["файл"], j) in old for j in (r["i"] - 1, r["i"], r["i"] + 1))))
        ch = {r["кусок"] for r in rows}
        badc = {r["кусок"] for r in rows if (r["файл"], r["i"]) in old}
        print(f"ПРЕЖНЯЯ (3.2, текст Whisper): поймано {sum(o for *_x, o in cau)/max(1,len(cau)):.0%}, "
              f"кусков без метки с подозрением {len(badc - lab_chunks)/max(1,len(ch - lab_chunks)):.1%}")
        print("     ", ", ".join(f"{i}@{f}:{'да' if o else 'нет'}" for i, f, o in cau))
    for lim in (0.05, 0.1, 0.15, 0.2, 0.3):
        c = [r for r in res if r[7] <= lim]
        if c:
            b = max(c, key=lambda r: (r[0], r[1]))
            print(f"ложных <= {lim:.0%}: поймано {b[0]:.0%}, ложных {b[7]:.1%} | post<{b[2]} или (post<{b[3]} и скелет<{b[4]}), пауза<{b[5]}, букв>={b[6]}")
            print("     ", ", ".join(f"{i}@{f}:{'да' if o else 'нет'}" for i, f, o in b[8]))
            print("      без метки:", ", ".join(b[9]))
    return res


if __name__ == "__main__":
    main()
