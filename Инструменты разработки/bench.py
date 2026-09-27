#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер скорости синтеза (RTF) на этой машине. Не часть приложения — инструмент.

RTF = секунды расчёта / секунды звука. Цель — существенно меньше 1.

    python bench.py --device cpu --nfe 16
    python bench.py --device mps --nfe 16
    python bench.py --engine f5mlx --nfe 8
"""
from __future__ import annotations

import os as _os, sys as _sys
# скрипт лежит в подпапке, а core/ — в корне проекта
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import os, sys, time, argparse
_hs = os.environ.get("PYTHONHASHSEED")
if _hs is None or not _hs.isdigit():
    os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from core.engines import get_engine
from core.normalize import normalize_text, sentenize
from core.stress import add_stress
from core.pipeline import _chunk_sentences

TEXT_FILE = "Примеры текстов/Пример — проба пера.txt"


def load_chunks(max_chars: int, n: int, stress: bool = True) -> list[str]:
    raw = open(TEXT_FILE, encoding="utf-8").read()
    t = normalize_text(raw, expand_numbers=True)
    if stress:
        t = add_stress(t)
    sents = sentenize(t)
    # берём из середины — там обычная проза, без заголовков
    mid = len(sents) // 3
    chunks = _chunk_sentences(" ".join(sents[mid:]), max_chars)
    return chunks[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="f5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--speed", type=float, default=1.55)
    ap.add_argument("--voice", default="голос диктора")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--chars", type=int, default=0, help="0 = как у движка")
    ap.add_argument("--out", default=None, help="куда записать wav для прослушивания")
    ap.add_argument("--tag", default="", help="метка в отчёте")
    ap.add_argument("--quant", type=int, default=0, help="MLX: 8 или 4 бита на вес")
    ap.add_argument("--dtype", default="float32", help="MLX: float32 | float16 | bfloat16")
    ap.add_argument("--schedule", default="epss", help="MLX: epss (прореженное) | uniform (как torch)")
    ap.add_argument("--total-sec", type=float, default=0, help="MLX: секунд (образец+кусок) на вызов")
    args = ap.parse_args()

    kw = {"speed": args.speed}
    if args.engine == "f5":
        kw["device"] = args.device
        kw["nfe_step"] = args.nfe
    else:
        kw["steps"] = args.nfe
        if args.quant:
            kw["quant_bits"] = args.quant
        if args.total_sec:
            kw["max_total_sec"] = args.total_sec
        kw["dtype"] = args.dtype
        kw["schedule"] = args.schedule
    eng = get_engine(args.engine, **kw)
    if args.chars:
        eng.max_chunk_chars = args.chars
    chunks = load_chunks(eng.max_chunk_chars, args.chunks)

    label = args.tag or (f"{args.engine}/{args.device}/nfe={args.nfe}/chars={eng.max_chunk_chars}"
                         + (f"/q{args.quant}" if args.quant else ""))
    print(f"### {label}", flush=True)
    t0 = time.time()
    eng.ensure_loaded()
    print(f"  загрузка модели: {time.time()-t0:.1f} c", flush=True)

    sr = eng.sample_rate
    pieces, rows = [], []
    for i, ch in enumerate(chunks, 1):
        t1 = time.time()
        a = eng.synth_chunk(ch, args.voice)
        dt = time.time() - t1
        dur = len(a) / sr
        rtf = dt / dur if dur > 0 else float("inf")
        rows.append((len(ch), dur, dt, rtf))
        print(f"  кусок {i}: {len(ch):4d} симв | звук {dur:6.2f} c | счёт {dt:7.2f} c | RTF {rtf:6.2f}",
              flush=True)
        if len(a):
            pieces.append(a)
            pieces.append(np.zeros(int(sr * 0.35), dtype=np.float32))

    if not pieces:
        print("  НЕТ ЗВУКА"); return 1
    audio = np.concatenate(pieces)
    tot_dur = len(audio) / sr
    tot_dt = sum(r[2] for r in rows)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    # первый кусок часто «прогревочный» — считаем и без него
    warm = rows[1:] or rows
    warm_rtf = sum(r[2] for r in warm) / sum(r[1] for r in warm)
    print(f"  ИТОГ: звук {tot_dur:.1f} c | счёт {tot_dt:.1f} c | RTF {tot_dt/tot_dur:.2f} | "
          f"RTF без 1-го куска {warm_rtf:.2f} | RMS {rms:.4f}", flush=True)
    if args.out:
        import soundfile as sf
        sf.write(args.out, audio, sr)
        print(f"  wav: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
