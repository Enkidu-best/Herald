#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скорость: RTF конвейера на кусках главы (Herald 3.1, раздел 3).

Один и тот же набор кусков (по умолчанию первые 12 кусков главы), кэш
выключен, проверка краёв включена - то есть так, как считается книга. Меняется
одно: размер пачки, упреждение, число шагов или голос (образец другой длины).

    python "Инструменты разработки/bench_speed.py" --text глава.txt --batch 1 --no-pipeline
    python "Инструменты разработки/bench_speed.py" --text глава.txt --batch 2
    python "Инструменты разработки/bench_speed.py" --text глава.txt --steps 12
    python "Инструменты разработки/bench_speed.py" --text глава.txt --voice "… (7 с)"

Квантование, float16 и два процесса здесь нарочно не меряются: проверено в
3.0 (README, «Скорость») - не помогают.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="текст главы (режется, как в программе)")
    ap.add_argument("--chunks-json", help="готовые куски (для сравнения с другой версией)")
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--out", default=None, help="сохранить звук (wav)")
    ap.add_argument("--no-retries", action="store_true",
                    help="без переозвучек: они случайны и забивают разницу в скорости")
    a = ap.parse_args()
    import mlx.core as mx
    import soundfile as sf
    from core.engines import get_engine
    from core.pipeline import _prepare_text, _chunk_paragraphs, _synth_chunks
    import core.pipeline as P
    if a.no_retries:
        P.TAIL_RETRIES = 0

    eng = get_engine("f5mlx", language="ru", steps=a.steps, schedule="epss")
    eng.ensure_loaded()
    if a.chunks_json:
        ch = [tuple(c) for c in json.load(open(a.chunks_json, encoding="utf-8"))][:a.chunks]
    else:
        text = _prepare_text(open(a.text, encoding="utf-8").read(), "+", True, True, "ru")
        ch = _chunk_paragraphs(text, eng.max_chunk_chars, 0.4, 0.8, 1.1)[:a.chunks]
    mx.random.seed(7)
    _synth_chunks(eng, a.voice, ch[:1], 24000, 0.4, use_cache=False)       # прогрев
    st: dict = {}
    t0 = time.time()
    audio = _synth_chunks(eng, a.voice, ch, 24000, 0.4, use_cache=False, stats=st,
                          batch_size=a.batch, pipelined=not a.no_pipeline)
    dt = time.time() - t0
    if a.out:
        sf.write(a.out, audio, 24000)
    res = {"пачка": a.batch, "упреждение": not a.no_pipeline, "шаги": a.steps,
           "голос": a.voice, "кусков": len(ch), "звук_с": round(st["fresh_sec"], 1),
           "счёт_с": round(dt, 1), "RTF": round(dt / max(st["fresh_sec"], 1e-6), 3),
           "переозвучек": st["retries"]}
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
