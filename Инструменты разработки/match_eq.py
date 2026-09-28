#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Автоподбор эквалайзера под живого диктора.

Идея вместо ручного тыканья в полосы: снять долгий средний спектр (LTAS) у
диктора и у синтеза, взять разницу — и она же будет кривой коррекции. Тембр
подтягивается к оригиналу целиком, без угадывания «а добавить ли на 7 кГц».

Два дня правок EQ на слух показали, почему это нужно: каждая ручная догадка
лечила одно и ломала другое. Разностная кривая не догадывается.

    python "Инструменты разработки/match_eq.py" ДИКТОР.wav СИНТЕЗ.wav
    python ... --limit 6 --smooth 2      # мягче: предел ±6 дБ, сильнее сглаживание

Печатает готовую строку фильтра для ffmpeg (firequalizer).
Зависит только от ffmpeg и numpy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

SR = 48000
# Сетка по третям октавы: достаточно подробно для тембра и не ловит форманты
# отдельных гласных (иначе кривая начнёт подгонять не голос, а конкретный текст).
BANDS = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                  10000, 12500, 16000], dtype=float)


def load(path: str, dur: float = 60.0) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-t", str(dur), "-i", path,
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def ltas(y: np.ndarray, n: int = 4096, hop: int = 2048) -> np.ndarray:
    """Долгий средний спектр по полосам — только по кадрам с речью.

    Тишина в расчёт не идёт: иначе паузы (которых в записях разное количество)
    сдвигают картину и кривая начинает исправлять не тембр, а монтаж.
    """
    if len(y) < n:
        return np.zeros(len(BANDS))
    w = np.hanning(n)
    frames = [y[i:i + n] for i in range(0, len(y) - n, hop)]
    rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    keep = rms > max(np.percentile(rms, 45), 1e-5)
    if not keep.any():
        keep = np.ones(len(frames), dtype=bool)
    spec = np.zeros(n // 2 + 1)
    used = 0
    for f, ok in zip(frames, keep):
        if ok:
            spec += np.abs(np.fft.rfft(f * w)) ** 2
            used += 1
    spec /= max(used, 1)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    out = np.zeros(len(BANDS))
    for i, fc in enumerate(BANDS):
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        m = (freqs >= lo) & (freqs < hi)
        out[i] = spec[m].mean() if m.any() else 1e-12
    return out


def curve(target: np.ndarray, source: np.ndarray, limit: float,
          smooth: int) -> np.ndarray:
    """Разница спектров в дБ: сколько добавить синтезу, чтобы стать диктором."""
    t = 10 * np.log10(target + 1e-12)
    s = 10 * np.log10(source + 1e-12)
    # Выравниваем по средней громкости: нас интересует ФОРМА спектра,
    # общий уровень потом выставит loudnorm.
    mid = (BANDS >= 200) & (BANDS <= 4000)
    diff = (t - t[mid].mean()) - (s - s[mid].mean())
    if smooth > 0:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        diff = np.convolve(np.pad(diff, smooth, mode="edge"), k, mode="valid")
    return np.clip(diff, -limit, limit)


def as_filter(gains: np.ndarray) -> str:
    entries = ";".join(f"entry({int(f)},{g:.1f})" for f, g in zip(BANDS, gains))
    return f"firequalizer=gain_entry='{entries}'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="запись живого диктора (цель)")
    ap.add_argument("source", help="синтез, который подтягиваем")
    ap.add_argument("--limit", type=float, default=8.0, help="предел коррекции, дБ")
    ap.add_argument("--smooth", type=int, default=1, help="сглаживание кривой")
    args = ap.parse_args()

    t = ltas(load(args.target))
    s = ltas(load(args.source))
    g = curve(t, s, args.limit, args.smooth)

    print("разностная кривая (сколько добавить синтезу):", file=sys.stderr)
    for f, v in zip(BANDS, g):
        bar = "+" * int(max(v, 0) * 2) + "-" * int(max(-v, 0) * 2)
        print(f"  {int(f):6d} Гц  {v:+5.1f} дБ  {bar}", file=sys.stderr)
    print(as_filter(g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
