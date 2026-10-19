#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Голос и ударения вместе (3.8, после оценки Андрея 19.10): замер на 10 фразах.

Для каждой папки-режима «<папка>/<режим>/<чтец>__NN.wav» (10 первых фраз набора
3.8, Архив/qwen голос 3.8/фразы.json):
  * похожесть на образец - ECAPA (speechbrain/spkrec-ecapa-voxceleb), среднее;
  * постоянство - похожесть записей режима между собой (держит ли голос);
  * ударения - meter2 в трудном слове фразы (из 10);
  * WER - доля ошибок в словах по Whisper large-v3-turbo.

    python "Инструменты разработки/measure_voice_stress.py" --dir "Архив/qwen голос 3.8" rs_now rs_rp105 ...
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import importlib.util
import json

import numpy as np


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Архив/qwen голос 3.8")
    ap.add_argument("modes", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import soundfile as sf
    sim = _mod("sim", "Инструменты разработки/bench_similarity_38.py")
    sim.KIND = "ecapa"
    eng = _mod("eng", "Инструменты разработки/bench_engines_38.py")
    from core import meter
    phr = json.load(open(_os.path.join(a.dir, "фразы.json"), encoding="utf-8"))
    voices = list(sim.VOICES)
    refs = {v: sim.emb(sim.load16(sim.ref_path(v))) for v in voices}
    out = {}
    print("| режим | похожесть на образец (Г / Я) | постоянство (Г / Я) | ударения верно | WER |")
    print("|---|---|---|---|---|")
    for mode in a.modes:
        S, P, ok, n, W = {}, {}, 0, 0, []
        for v in voices:
            E = []
            for i, p in enumerate(phr):
                f = _os.path.join(a.dir, mode, f"{v}__{i + 1:02d}.wav")
                if not _os.path.exists(f):
                    continue
                x = sim.load16(f)
                if len(x) < 8000:
                    W.append(1.0)
                    n += 1
                    continue
                E.append(sim.emb(x))
                au, sr = sf.read(f, dtype="float32")
                if p["word"]:
                    bad = meter.wrong_words(au, sr, p["marked"],
                                            only={p["word"].replace("+", "").lower()}, min_margin=0.0)
                    ok += not bad
                    n += 1
                W.append(eng.wer_of(au, sr, p["phrase"])[0])
            S[v] = float(np.mean([e @ refs[v] for e in E])) if E else float("nan")
            P[v] = sim.pair_mean(E)
        out[mode] = {"sim": S, "pair": P, "stress": [ok, n], "wer": float(np.mean(W)) if W else None}
        g, y = voices
        print(f"| {mode} | {S[g]:.2f} / {S[y]:.2f} | {P[g]:.2f} / {P[y]:.2f} | {ok} из {n} | "
              f"{np.mean(W):.1%} |", flush=True)
    json.dump(out, open(a.out or _os.path.join(a.dir, "замер.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
