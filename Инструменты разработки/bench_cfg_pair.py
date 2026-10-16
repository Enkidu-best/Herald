#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пара для слепого сравнения (3.5, п. 1): подсказка CFG на всех шагах и на первых K.

Один отрывок, одни и те же зёрна на каждом куске, без пересчётов и проверок -
различие в паре только в CFG. Пишет два wav и время счёта каждого.

    python "Инструменты разработки/bench_cfg_pair.py" --cfg 8 --out-dir пара/
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--paras", type=int, default=3)
    ap.add_argument("--seed", type=int, default=33)
    a = ap.parse_args()
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    import core.pipeline as P
    from core import profiles
    from core.engines import get_engine, clone_engine
    src = open("Примеры текстов/Пример - проба пера.txt", encoding="utf-8").read()
    paras = [p for p in src.split("\n\n") if p.strip()][:a.paras + 1]
    text = P._prepare_text("\n\n".join(paras), "+", True, True, "ru")
    _os.makedirs(a.out_dir, exist_ok=True)
    for name, k in (("все", None), (f"первые_{a.cfg}", a.cfg)):
        eng = get_engine(clone_engine(), language="ru", cfg_steps=k, **profiles.get(None).kwargs)
        eng.ensure_loaded()
        eng.synth_batch(["Пров+ерка."], a.voice)
        chunks = P._chunk_paragraphs(text, eng.max_chunk_chars, 0.4, 0.8, 1.1)
        pieces, t0 = [], time.monotonic()
        for ci, c in enumerate(chunks):
            mx.random.seed(a.seed + ci)
            w = eng.synth_batch([P._with_end_punct(c[0])], a.voice, [c[2]])[0]
            pieces.append((w, c[1]))
        took = time.monotonic() - t0
        audio = P._join(pieces, 24000)
        path = _os.path.join(a.out_dir, f"cfg_{name}.wav")
        sf.write(path, audio, 24000)
        print(f"{name}: звук {len(audio) / 24000:.1f} с, счёт {took:.1f} с, "
              f"RTF {took / (len(audio) / 24000):.2f} -> {path}", flush=True)
        del eng
        P.unload_engines()
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
