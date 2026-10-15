#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Глава: время и слова «под подозрением» по звуку (3.3, пп. 1-2, 4).

Озвучивает текст с одним зерном и считает, сколько слов итогового звука
проверка по звуку считает съеденными или смазанными (core/align.is_bad):
    --check none   - как 3.2 без проверки (только края);
    --check sound  - 3.3: проверка по звуку с пересчётом;
    --max-chars N  - длина куска (по умолчанию как в программе, 280).

    python "Инструменты разработки/bench_sound.py" --text отрывок.txt --check none --out а.wav
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
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--check", choices=["none", "sound", "strict"], default="none")
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import mlx.core as mx
    import core.pipeline as P
    from core import profiles, align
    from core.engines import get_engine, clone_engine
    eng = get_engine(clone_engine(), language="ru", **profiles.get(None).kwargs)
    eng.ensure_loaded()
    al = align.aligner().load()
    mc = a.max_chars or eng.max_chunk_chars
    texts = [P._prepare_text(open(t, encoding="utf-8").read(), "+", True, True, "ru") for t in a.text]
    pieces = []
    total_t, total_a, st = 0.0, 0.0, {}
    parts = []
    for t in texts:
        ch = P._chunk_paragraphs(t, mc, 0.4, 0.8, 1.1)
        mx.random.seed(a.seed)
        t0 = time.time()
        audio = P._synth_chunks(eng, a.voice, ch, 24000, 0.4, use_cache=False, stats=st,
                                strict_words=(a.check == "strict"),
                                sound_check=(a.check == "sound"),
                                on_piece=lambda tx, au: pieces.append((tx, au)))
        total_t += time.time() - t0
        total_a += len(audio) / 24000
        parts += [audio, np.zeros(24000, np.float32)]
    sf.write(a.out, np.concatenate(parts), 24000)
    # слова под подозрением в ИТОГОВОМ звуке каждого куска
    bad, words, worst = [], 0, []
    t0 = time.time()
    for tx, au in pieces:
        ws = al.score(au, 24000, tx)
        words += len(ws)
        bad += [w.word for w in ws if align.is_bad(w)]
        if ws:
            worst.append(min(w.post_mean for w in ws))
    t_align = time.time() - t0
    res = {"проверка": a.check, "длина_куска": mc, "кусков": len(pieces),
           "звук_с": round(total_a, 1), "счёт_с": round(total_t, 1),
           "RTF": round(total_t / max(total_a, 1), 3), "пересинтезов": st.get("retries", 0),
           "кусков_с_пересинтезом": st.get("retry_chunks", 0),
           "слов": words, "под_подозрением": len(bad), "слова": bad,
           "кусков_с_подозрением": sum(1 for (tx, au) in pieces
                                       if any(align.is_bad(w) for w in al.score(au, 24000, tx))),
           "выравнивание_с_на_кусок": round(t_align / max(1, len(pieces)), 2)}
    json.dump(res, open(a.out[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
