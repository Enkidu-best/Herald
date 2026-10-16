#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Режимы «Качество / скорость» (3.6, п. 1): звук, прогноз, факт - как окно.

На каждый режим: книга-образец (--learn) озвучивается целиком - из неё, как в
окне, берутся ожидание на секунду звука и поправка длины звука голоса; затем
книга --book: прогноз «с нуля» (прикидка режима, как у голоса без замеров),
прогноз по образцу и факт.

    python "Инструменты разработки/bench_modes.py" --voice "Игорь Ященко" --out режимы.json
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import shutil
import tempfile
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Игорь Ященко")
    ap.add_argument("--book", default="Примеры текстов/Пример - проба пера 3.txt")
    ap.add_argument("--learn", default="Примеры текстов/Пример - проба пера 2.txt")
    ap.add_argument("--modes", default="balanced,fast,max")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import core.pipeline as P
    from core import profiles
    from core.engines import clone_engine
    from core.voicefx import load as load_fx
    from core.engines.f5 import default_voices_dir
    import tts_gui as G
    rate = load_fx(default_voices_dir(), a.voice).speed or 1.0
    main_p = profiles.get(None)

    def run(path, kw):
        out = tempfile.mkdtemp(prefix="herald_modes_")
        t0 = time.monotonic()
        res = P.convert_book(path, engine=clone_engine(), eng_kwargs=dict(kw), voice=a.voice,
                             out_dir=out, options=P.ConvertOptions(use_cache=False))
        took = time.monotonic() - t0
        shutil.rmtree(out, ignore_errors=True)
        return res.duration, took

    rows = []
    for q in a.modes.split(","):
        kw = profiles.engine_kwargs(main_p, q)
        P.unload_engines()
        P._loaded_engine(clone_engine(), dict(kw, language="ru"))   # модель загружена, как в окне
        e_l, e_b = P.estimate_book(a.learn, 15.0), P.estimate_book(a.book, 15.0)
        cold = e_b.audio_sec / rate * profiles.rtf_for(main_p, "mlx", q) * G.SOUND_COST
        d_l, t_l = run(a.learn, kw)
        k, w = d_l / (e_l.audio_sec / rate), t_l / d_l
        learned = e_b.audio_sec / rate * k * w
        d_b, t_b = run(a.book, kw)
        row = {"режим": profiles.quality(q).title, "cfg_steps": kw.get("cfg_steps"),
               "звук_с": round(d_b, 1), "прогноз_с_нуля_с": round(cold, 1),
               "прогноз_по_книге_с": round(learned, 1), "факт_с": round(t_b, 1),
               "ожидание_на_с_звука": round(t_b / d_b, 2),
               "ошибка_прогноза_%": round(100 * (learned - t_b) / t_b, 1)}
        rows.append(row)
        print("   ", row, flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
