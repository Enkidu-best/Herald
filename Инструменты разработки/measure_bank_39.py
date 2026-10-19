#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11: замер голосов (20 фраз): похожесть на свой образец (ECAPA),
постоянство (записи между собой), ударения meter2 (17 трудных слов), WER (Whisper).

    python "Инструменты разработки/measure_bank_39.py" native icl_fixed ...
Режимы - папки Архив/голоса диктора 3.9/<режим>/<метка>__NN.wav; образец метки -
чтец (папка голосов Herald) или «Голос NN» (Архив/голоса диктора 3.9/образцы/).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import importlib.util
import json

import numpy as np

BANK = "Архив/голоса диктора 3.9"


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    import soundfile as sf
    sim = _mod("sim", "Инструменты разработки/bench_similarity_38.py")
    sim.KIND = "ecapa"
    eng = _mod("eng", "Инструменты разработки/bench_engines_38.py")
    from core import meter
    setname = _os.environ.get("VOICE_SET", "куски")
    phr = json.load(open(_os.path.join(BANK, setname + ".json"), encoding="utf-8"))
    for p in phr:                                  # «фразы»: одно слово; «куски»: несколько
        p.setdefault("words", [p["word"]] if p.get("word") else [])
    path = _os.path.join(BANK, "замер.json")
    res = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    print("| режим | голос | похожесть на образец | постоянство | ударения (трудные слова) | WER | пустых |")
    print("|---|---|---|---|---|---|---|")
    for mode in _sys.argv[1:]:
        labels = sorted({_os.path.basename(f).split("__")[0]
                         for f in glob.glob(_os.path.join(BANK, mode, "*__*.wav"))})
        for lab in labels:
            ref = (sim.ref_path(lab) if lab in sim.VOICES
                   else _os.path.join(BANK, "образцы", lab + ".wav"))
            r = sim.emb(sim.load16(ref))
            E, ok, n, W, empty = [], 0, 0, [], 0
            for i, p in enumerate(phr):
                f = _os.path.join(BANK, mode, f"{lab}__{i + 1:02d}.wav")
                au, sr = sf.read(f, dtype="float32")
                if len(au) < sr // 2:
                    empty += 1
                    W.append(1.0)
                    n += len(p["words"])
                    continue
                E.append(sim.emb(sim.load16(f)))
                if p["words"]:
                    want = {w.replace("+", "").lower() for w in p["words"]}
                    bad = meter.wrong_words(au, sr, p["marked"], only=want, min_margin=0.0)
                    bad_set = {b.replace("+", "").lower() for b in bad}
                    ok += len(want - bad_set)
                    n += len(want)
                W.append(eng.wer_of(au, sr, p["phrase"])[0])
            s = float(np.mean([e @ r for e in E])) if E else float("nan")
            pc = sim.pair_mean(E)
            res[f"{mode}|{lab}"] = {"sim": s, "pair": pc, "stress": [ok, n], "wer": float(np.mean(W)),
                                    "empty": empty}
            print(f"| {mode} | {lab} | {s:.2f} | {pc:.2f} | {ok} из {n} | {np.mean(W):.1%} | {empty} |",
                  flush=True)
            json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
