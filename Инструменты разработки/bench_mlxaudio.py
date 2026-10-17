#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер моделей mlx-audio на этом Mac (3.8, задание 10а, п. 1.5): во сколько раз
быстрее реального времени, память, время первого звука.

Тяжёлый шаг - под ../.heavy-lock. Модели - в отдельном процессе (.venv-mlxaudio).

    python "Инструменты разработки/bench_mlxaudio.py" --models qwen-stress:8,qwen-stress:4,chatterbox,voxcpm2
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import time

TEXT = ("Смотритель достал из шкафчика старую подзорную трубу, протёр окуляр рукавом и "
        "навёл её на отмель. Белый свет стоял там же, где и вчера, - ровный, без мигания.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen-stress:8,qwen-stress:4,chatterbox,voxcpm2")
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="Архив/mlx-audio 3.8/замер.json")
    a = ap.parse_args()
    import core.pipeline as P
    from core.engines.mlxaudio import MlxAudioEngine, MODELS
    _os.makedirs(_os.path.dirname(a.out), exist_ok=True)
    rows = []
    for spec in a.models.split(","):
        name, _, bits = spec.partition(":")
        eng = MlxAudioEngine(model=name, bits=int(bits or 8))
        text = P._prepare_text(TEXT, eng.stress_format, True, True, "ru")
        t0 = time.monotonic()
        try:
            eng.ensure_loaded()
        except Exception as e:
            rows.append({"model": spec, "error": str(e)})
            print(spec, "не загрузилась:", e, flush=True)
            continue
        load = time.monotonic() - t0
        eng.synth_chunk("Пров+ерка.", a.voice, seed=1)            # прогрев
        rtf, ttfb, peak = [], [], []
        for i in range(a.repeat):
            eng.synth_chunk(text, a.voice, seed=10 + i)
            s = eng.last_stats
            rtf.append(s["wall"] / max(s["audio_sec"], 1e-3))
            peak.append(s["peak_gb"])
            eng.synth_chunk(text, a.voice, seed=10 + i, stream=True)
            ttfb.append(eng.last_stats["ttfb"])
        row = {"model": spec, "title": MODELS[name]["title"], "sr": eng.sample_rate,
               "load_sec": round(load, 1), "mem_gb": eng.load_stats.get("mem_gb"),
               "peak_gb": max(peak), "rtf": round(sorted(rtf)[len(rtf) // 2], 3),
               "x_realtime": round(1 / sorted(rtf)[len(rtf) // 2], 2),
               "ttfb": round(sorted(ttfb)[len(ttfb) // 2], 2)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        eng.close()
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
