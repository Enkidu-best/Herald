#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, п. 2: есть ли паузы между предложениями (Андрей: «предложения
наезжают друг на друга»). Тишина - громкость ниже -40 дБ от пика, не короче 0,15 с.
Печатает: пауз на минуту, медиану, долю пауз 0,3-0,6 с (после предложения),
0,6-1,0 (после абзаца) и > 1,0 с (между репликами).

    python "Инструменты разработки/measure_pauses_39.py" файл.mp3 ...
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np


def pauses(path: str, sr: int = 16000) -> list[float]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)
    fr = int(0.01 * sr)
    rms = np.sqrt(np.mean(a[: len(a) // fr * fr].reshape(-1, fr) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms / rms.max())
    quiet = db < -40
    out, run = [], 0
    for q in quiet:
        if q:
            run += 1
        else:
            if run * 0.01 >= 0.15:
                out.append(run * 0.01)
            run = 0
    return out[1:] if out else out          # первая - тишина в начале файла


def main() -> int:
    print("| файл | минут | пауз в минуту | медиана, с | 0,3-0,6 с | 0,6-1,0 с | > 1,0 с |")
    print("|---|---|---|---|---|---|---|")
    for f in sys.argv[1:]:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "csv=p=0", f], capture_output=True, text=True).stdout)
        p = np.array(pauses(f))
        n = len(p)
        print(f"| {f.split('/')[-1]} | {dur / 60:.1f} | {n / (dur / 60):.0f} | "
              f"{np.median(p) if n else 0:.2f} | {np.mean((p >= .3) & (p < .6)) if n else 0:.0%} | "
              f"{np.mean((p >= .6) & (p < 1.0)) if n else 0:.0%} | {np.mean(p >= 1.0) if n else 0:.0%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
