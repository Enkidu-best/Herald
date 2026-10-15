#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Порог проверки слов по звуку - по размеченному набору (3.3).

Вход - JSON от eval_align.py. Слово «под подозрением», если его оценка ниже
порога; место Андрея «поймано», если под подозрением само слово или соседнее
(выравнивание может сдвинуть смазанный звук на соседа); «слитно» - если пауза
после слова на знаке препинания короче порога. Ложное срабатывание - кусок,
где есть подозрение, но Андрей ничего не отмечал.

Цель задания: поймать >= 80 % мест при ложных <= 5 % кусков.

    python "Инструменты разработки/eval_align_threshold.py" слова.json [--skip id ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def evaluate(rows, feat, thr, min_letters, gap_thr, skip):
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["файл"]].append(r)
    flagged = set()
    for r in rows:
        if r["букв"] < min_letters:
            continue
        v = r[feat]
        bad = v > thr if feat == "low" else v < thr
        merged = gap_thr > 0 and r["знак"] in ",.!?…:;—–" and r["gap"] < gap_thr
        if bad or merged:
            flagged.add((r["файл"], r["i"]))
    labels, caught = [], []
    labeled_chunks = set()
    for r in rows:
        for lab in r.get("метки", []):
            if lab["id"] in skip:
                continue
            labels.append((r["файл"], r["i"], lab))
            labeled_chunks.add(r["кусок"])
    for f, i, lab in labels:
        if lab["вид"] == "слитно":
            r = next(x for x in by_file[f] if x["i"] == i)
            ok = gap_thr > 0 and r["gap"] < gap_thr
        else:
            ok = any((f, j) in flagged for j in (i - 1, i, i + 1))
        caught.append((lab["id"], f, ok))
    chunks = {r["кусок"] for r in rows}
    bad_chunks = {r["кусок"] for r in rows if (r["файл"], r["i"]) in flagged}
    fp = len(bad_chunks - labeled_chunks) / max(1, len(chunks - labeled_chunks))
    rec = sum(ok for _l, _f, ok in caught) / max(1, len(caught))
    evaluate.last_fp_chunks = sorted(bad_chunks - labeled_chunks)
    return rec, fp, caught, len(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--feat", default=None)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--ml", type=int, default=1)
    ap.add_argument("--gap", type=float, default=0.0)
    a = ap.parse_args()
    rows = json.load(open(a.rows, encoding="utf-8"))
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from eval_align import apply_labels
    apply_labels(rows, json.load(open(os.path.join(here, "разметка_3.3", "набор.json"), encoding="utf-8")))
    print("слов", len(rows), "| места Андрея:")
    for r in rows:
        for lab in r.get("метки", []):
            nb = [x for x in rows if x["файл"] == r["файл"] and abs(x["i"] - r["i"]) <= 1]
            print(f"   {lab['id']:12s} {r['файл']:10s} " + "  ".join(
                f"{x['слово']}(post {x.get('post',0):.2f} pmin {x.get('pmin',0):.2f} sim {x.get('sim',0):.2f} «{x.get('слышно','')}» пауза {x['gap']:.2f})"
                for x in nb))
    clean = [r for r in rows if not r.get("метки")]
    for k in ("post", "pmin", "sim"):
        v = [r.get(k, 0) for r in clean]
        print(f"чистые слова: {k} медиана {np.median(v):.2f}, 5-й процентиль {np.percentile(v, 5):.2f}, 1-й {np.percentile(v, 1):.2f}")
    best = []
    for feat, grid in (("gop", np.arange(0.20, 0.80, 0.02)), ("min", np.arange(0.02, 0.50, 0.02)),
                       ("low", np.arange(0.2, 0.9, 0.05)), ("post", np.arange(-4.0, -0.1, 0.1)),
                       ("pmin", np.arange(0.001, 0.5, 0.01)), ("sim", np.arange(0.2, 0.95, 0.05))):
        for ml in (1, 2, 3, 4):
            for g in (0.0, 0.03, 0.05, 0.08, 0.12):
                for t in grid:
                    rec, fp, caught, n = evaluate(rows, feat, float(t), ml, g, set(a.skip))
                    best.append((rec, -fp, feat, round(float(t), 2), ml, g, fp, caught, n))
    ok = [b for b in best if b[6] <= 0.05]
    ok.sort(key=lambda b: (b[0], b[1]), reverse=True)
    print("\nЛучшее при ложных <= 5 % кусков:")
    for b in ok[:8]:
        rec, _, feat, t, ml, g, fp, caught, n = b
        print(f"  {feat} порог {t}  букв >= {ml}  пауза < {g}: поймано {rec:.0%}, ложных {fp:.1%} кусков (из {n})")
    if ok:
        b = ok[0]
        print("  поймано/пропущено:", ", ".join(f"{i}@{f}:{'да' if o else 'нет'}" for i, f, o in b[7]))
    print("\nПо выбранному порогу (--feat/--thr): что срабатывает без метки")
    if a.feat:
        rec, fp, caught, n = evaluate(rows, a.feat, a.thr, a.ml, a.gap, set(a.skip))
        print(f"  поймано {rec:.0%}, кусков без метки с подозрением {fp:.1%} из {n}")
        print("  поймано/пропущено:", ", ".join(f"{i}@{f}:{'да' if o else 'нет'}" for i, f, o in caught))
        for ch in evaluate.last_fp_chunks:
            ws = [r for r in rows if r["кусок"] == ch and (r[a.feat] < a.thr if a.feat != "low" else r[a.feat] > a.thr) and r["букв"] >= a.ml]
            print(f"   {ch}: " + ", ".join(f"{r['слово']} «{r.get('слышно','')}» ({r[a.feat]:.2f}, {r['t']:.0f} с)" for r in ws))
    print("\nРазмен (лучший по каждому уровню ложных):")
    for lim in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        c = [b for b in best if b[6] <= lim]
        if c:
            b = max(c, key=lambda b: (b[0], b[1]))
            print(f"  ложных <= {lim:.0%}: поймано {b[0]:.0%} ({b[2]} {b[3]}, букв >= {b[4]}, пауза < {b[5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
