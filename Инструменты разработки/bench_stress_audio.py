#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ударения в звуке по режимам (3.7, п. 1.3): слепой набор для Андрея.

30 фраз (ударения_30.txt) в режимах «Быстрее» / «Сбалансировано» / «Максимальное
качество», двумя голосами. Одно и то же зерно на фразу во всех режимах, без
пересчётов и проверок - разница только в режиме. Каждая фраза - отдельный wav;
первые 10 фраз каждого режима склеены в один файл (пауза 1,5 с) - его и слушать.

    python "Инструменты разработки/bench_stress_audio.py" --out-dir ПАПКА
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voices", default="Вячеслав Герасимов,Игорь Ященко")
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    import core.pipeline as P
    from core import profiles
    from core.engines import get_engine, clone_engine
    rows = [l.split("|") for l in open("Инструменты разработки/ударения_30.txt", encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    phrases = [r[0].strip() for r in rows]
    texts = [P._with_end_punct(P._prepare_text(p, "+", True, True, "ru")) for p in phrases]
    _os.makedirs(a.out_dir, exist_ok=True)
    json.dump([{"фраза": p, "слово": r[1].strip(), "верно": r[2].strip(), "в модель": t}
               for p, r, t in zip(phrases, rows, texts)],
              open(_os.path.join(a.out_dir, "фразы.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for q in ("fast", "balanced", "max"):
        eng = get_engine(clone_engine(), language="ru",
                         **profiles.engine_kwargs(profiles.get(None), q))
        eng.ensure_loaded()
        for voice in a.voices.split(","):
            eng.synth_batch(["Пров+ерка."], voice)
            t0 = time.monotonic()
            first10 = []
            for i, t in enumerate(texts):
                mx.random.seed(a.seed + i)
                w = eng.synth_batch([t], voice)[0]
                sf.write(_os.path.join(a.out_dir, f"{voice}__{q}__{i + 1:02d}.wav"), w, 24000)
                if i < 10:
                    first10 += [w, np.zeros(int(1.5 * 24000), np.float32)]
            sf.write(_os.path.join(a.out_dir, f"{voice}__{q}__10 фраз.wav"),
                     np.concatenate(first10), 24000)
            print(f"{q} / {voice}: {time.monotonic() - t0:.0f} с", flush=True)
        del eng
        P.unload_engines()
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
