#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Мини-плеер (3.2): оба способа - AVAudioPlayer (Mac) и ffmpeg+sounddevice
(Windows, Linux) - на тишине: играть, позиция идёт, пауза держит позицию,
перемотка, стоп возвращает в начало. Звук не слышен - файл из нулей.

    python "Инструменты разработки/test_player.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import os
import sys
import tempfile
import time

fails: list[str] = []


def must(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        fails.append(msg)


def main() -> int:
    import numpy as np
    import soundfile as sf
    import player as P
    d = tempfile.mkdtemp(prefix="herald_fx_")
    f = os.path.join(d, "тишина.wav")
    sf.write(f, np.zeros(24000 * 6, dtype=np.float32), 24000)
    kinds = ([P._AVPlayer] if sys.platform == "darwin" else []) + [P._FFPlayer]
    for kind in kinds:
        print(kind.__name__ + ":")
        p = kind(f)
        must(abs(p.duration - 6.0) < 0.1, f"длительность {p.duration:.2f}")
        p.play(); time.sleep(1.2)
        must(p.playing and 0.6 < p.position < 1.8, f"играет, позиция {p.position:.2f}")
        p.pause(); pos = p.position; time.sleep(0.5)
        must(not p.playing and abs(p.position - pos) < 0.05, f"пауза держит {p.position:.2f}")
        p.seek(4.0)
        must(abs(p.position - 4.0) < 0.1, f"перемотка на 4 с: {p.position:.2f}")
        p.play(); time.sleep(0.6)
        must(p.playing and p.position > 4.2, f"играет с 4 с: {p.position:.2f}")
        p.stop()
        must(not p.playing and p.position < 0.05, "стоп - в начало")
    w = P.Player()
    must(w.load(f) and w.duration > 5, "Player.load выбирает рабочий способ")
    must(not P.Player().load(os.path.join(d, "нет.mp3")), "нет файла - False")
    print("\nOK: плеер" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
