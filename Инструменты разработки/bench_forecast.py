#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогноз ожидания против факта (3.5, п. 1): так же, как окно.

Книга 1 озвучивается целиком - запоминаются полное ожидание на секунду звука
и поправка длины звука голоса (tts_gui._learn_book). По ним предсказывается
книга 2, она озвучивается, прогноз сверяется с фактом (цель - в пределах 10 %).

    python "Инструменты разработки/bench_forecast.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import tempfile
import time


def run(P, path, voice, kw):
    from core.engines import clone_engine
    out = tempfile.mkdtemp(prefix="herald_fc_")
    t0 = time.monotonic()
    res = P.convert_book(path, engine=clone_engine(), eng_kwargs=dict(kw), voice=voice,
                         out_dir=out, options=P.ConvertOptions(use_cache=False))
    took = time.monotonic() - t0
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    return res.duration, took


def main() -> int:
    import core.pipeline as P
    from core import profiles
    from core.voicefx import load as load_fx
    from core.engines.f5 import default_voices_dir
    voice = "Вячеслав Герасимов"
    kw = dict(profiles.get(None).kwargs)
    rate = load_fx(default_voices_dir(), voice).speed or 1.0
    b1, b2 = "Примеры текстов/Пример - проба пера.txt", "Примеры текстов/Пример - проба пера 2.txt"
    P._loaded_engine("f5mlx", dict(kw, language="ru"))        # модель уже загружена, как в окне
    e1 = P.estimate_book(b1, 15.0)
    d1, t1 = run(P, b1, voice, kw)
    k = d1 / (e1.audio_sec / rate)
    w = t1 / d1
    e2 = P.estimate_book(b2, 15.0)
    pred_audio = e2.audio_sec / rate * k
    pred = pred_audio * w
    d2, t2 = run(P, b2, voice, kw)
    print(f"книга 1: звук {d1:.1f} с, ожидание {t1:.1f} с (на с звука {w:.2f}), поправка длины {k:.3f}")
    print(f"книга 2: прогноз звука {pred_audio:.1f} с против {d2:.1f} с; "
          f"прогноз ожидания {pred:.1f} с против {t2:.1f} с - расхождение {100 * (pred - t2) / t2:+.0f} %")
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
