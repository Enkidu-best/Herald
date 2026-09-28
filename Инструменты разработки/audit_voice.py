#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Перцептивный аудит голоса: метрики, которые ловят «металл», «бочку»,
пере-бас и сжатую динамику — то, что ASR-разборчивость и пик НЕ ловят.

Идея: не гнаться за абсолютными порогами, а мерить ОТКЛОНЕНИЕ от живого
диктора (того самого образца, из которого клонирован голос). Диктор — это и
есть цель; чем ближе к нему по этим четырём числам, тем «богаче/натуральнее».

Запуск:
    python3 audit_voice.py ЦЕЛЬ.mp3 кандидат1.mp3 [кандидат2.mp3 ...]
где ЦЕЛЬ — фрагмент живого диктора. Печатает таблицу и дельты.
Можно как гейт в тесте: возвращает код !=0, если кандидат заметно хуже цели
по «металлу» или «бочке».

Зависит только от ffmpeg (декод) и numpy.
"""
import subprocess, sys, struct, numpy as np

SR = 48000

def load(path, offset=3.0, dur=40.0):
    """Декод в float32 mono 48k через ffmpeg (без librosa/soundfile)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset), "-t", str(dur),
           "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32).copy()
    if y.size:
        y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def _stft(y, n=2048, hop=512):
    w = np.hanning(n)
    cols = 1 + (len(y) - n) // hop if len(y) >= n else 0
    out = np.empty((n // 2 + 1, max(cols, 0)), dtype=np.float64)
    for i in range(cols):
        seg = y[i * hop:i * hop + n] * w
        out[:, i] = np.abs(np.fft.rfft(seg)) ** 2
    return out

def metal(y):
    """Спектральная плоскость в 4-11 кГц. Выше = больше шумового «песка»
    вокодера = «металл». У чистой речи верх гармонический (низкая плоскость)."""
    S = _stft(y); f = np.fft.rfftfreq(2048, 1 / SR)
    Sb = S[(f >= 4000) & (f < 11000), :] + 1e-10
    if Sb.shape[1] == 0:
        return float("nan")
    gm = np.exp(np.mean(np.log(Sb), 0)); am = np.mean(Sb, 0)
    return float(np.median(gm / am))

def boxy(y):
    """Пик-к-медиане в 300-700 Гц. Узкий резонанс = «бочка/гул»."""
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))); f = np.fft.rfftfreq(len(y), 1 / SR)
    b = S[(f >= 300) & (f < 700)]
    return float(np.max(b) / (np.median(b) + 1e-9))

def bass_pct(y):
    """Доля энергии 60-250 Гц. Сильно выше диктора = бубнит/«бочка»."""
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2; f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(100 * S[(f >= 60) & (f < 250)].sum() / (S.sum() + 1e-12))

def dynamic(y):
    """Разброс громкости по кадрам (95-й минус 10-й перцентиль), дБ.
    Низко = пересжато лимитером = «плоско, неживо»."""
    n = int(0.4 * SR); hop = int(0.2 * SR)
    cols = 1 + (len(y) - n) // hop if len(y) >= n else 0
    r = np.array([np.sqrt(np.mean(y[i*hop:i*hop+n] ** 2)) for i in range(cols)]) + 1e-9
    r = r[r > r.max() * 0.05]
    return float(np.percentile(20*np.log10(r), 95) - np.percentile(20*np.log10(r), 10))

def measure(path):
    y = load(path)
    return dict(metal=metal(y), boxy=boxy(y), bass=bass_pct(y), dyn=dynamic(y))

def main():
    if len(sys.argv) < 3:
        print("usage: audit_voice.py ЦЕЛЬ.mp3 кандидат.mp3 [...]"); return 2
    tgt = measure(sys.argv[1])
    print(f"{'':26s} {'металл':>7} {'бочка':>6} {'бас%':>6} {'динамика':>9}")
    print(f"{'ЦЕЛЬ (живой диктор)':26s} {tgt['metal']:7.3f} {tgt['boxy']:6.1f} "
          f"{tgt['bass']:6.1f} {tgt['dyn']:8.1f}д")
    worst = 0.0
    for p in sys.argv[2:]:
        m = measure(p)
        dm = m['metal'] - tgt['metal']; db = m['boxy'] - tgt['boxy']
        name = p.rsplit("/", 1)[-1][:24]
        print(f"{name:26s} {m['metal']:7.3f} {m['boxy']:6.1f} {m['bass']:6.1f} "
              f"{m['dyn']:8.1f}д   Δметалл {dm:+.3f}  Δбочка {db:+.1f}")
        worst = max(worst, dm)
    # гейт: металл заметно выше цели — регрессия «железного звука»
    return 1 if worst > 0.05 else 0

if __name__ == "__main__":
    raise SystemExit(main())
