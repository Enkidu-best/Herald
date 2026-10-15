#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пропавшие и неразборчивые слова в середине куска: «было -> стало» (3.2, п. 1.3).

Один и тот же набор текстов озвучивается с одинаковым зерном дважды:
  --mode old  - как 3.1: проверяются только края куска (Whisper small);
  --mode new  - строгая проверка всего куска (Whisper large-v3-turbo), до двух
                пересинтезов с другим зерном, затем лучший дубль.
Судья - Whisper large-v3 (НЕ turbo, чтобы не мерить проверку ею же): сколько
словарных слов текста (от 4 букв) пропало или искажено по последним буквам.

    python "Инструменты разработки/bench_words.py" --mode new --out ПАПКА \\
        --text "отрывок.txt" --items входы.json:2_диалог [--extra-dialog]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import contextlib
import io
import json
import os
import tempfile
import time

SR = 24000
JUDGE = "mlx-community/whisper-large-v3-mlx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", action="append", default=[], help="файл книги/главы")
    ap.add_argument("--items", default=None, help="входы.json:id1,id2")
    ap.add_argument("--extra-dialog", action="store_true", help="диалог из test_pauses")
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--seed", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import numpy as np
    import soundfile as sf
    import mlx.core as mx
    import mlx_whisper
    import core.pipeline as P
    from core.engines import get_engine
    from core.readers import read_document
    from core.chapters import build_chapters

    texts = []
    for f in a.text:
        texts.append((os.path.splitext(os.path.basename(f))[0],
                      P._prepare_text(build_chapters(read_document(f), 60)[0].text, "+", True, True, "ru")))
    if a.items:
        path, ids = a.items.split(":")
        for it in json.load(open(path, encoding="utf-8")):
            if it["id"] in ids.split(","):
                texts.append((it["id"], it["stressed"]))
    if a.extra_dialog:
        sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        _sys.path.insert(0, sys_path)
        from test_pauses import DIALOG_TEXT
        texts.append(("диалог_тест", P._prepare_text(DIALOG_TEXT, "+", True, True, "ru")))

    eng = get_engine("f5mlx", language="ru", steps=12, schedule="epss")
    eng.ensure_loaded()

    def judge(audio, text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SR)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = mlx_whisper.transcribe(f.name, path_or_hf_repo=JUDGE, language="ru",
                                           verbose=False, condition_on_previous_text=False)
        finally:
            os.unlink(f.name)
        heard = [(w, 0, 0) for w in P._words(r["text"])]
        return P._bad_words(heard, text)

    res = {"режим": a.mode, "тексты": {}}
    total_t, total_sec = 0.0, 0.0
    agg = {"кусков": 0, "кусков_с_пересинтезом": 0, "пересинтезов": 0}
    for name, text in texts:
        ch = P._chunk_paragraphs(text, eng.max_chunk_chars, 0.4, 0.8, 1.1)
        mx.random.seed(a.seed)
        st: dict = {}
        t0 = time.time()
        audio = P._synth_chunks(eng, a.voice, ch, SR, 0.4, use_cache=False, stats=st,
                                strict_words=(a.mode == "new"))
        dt = time.time() - t0
        sf.write(os.path.join(a.out, f"{name}.wav"), audio, SR)
        full = " ".join(c[0] for c in ch)
        bad = judge(audio, full)
        res["тексты"][name] = {"кусков": len(ch), "звук_с": round(len(audio) / SR, 1),
                               "счёт_с": round(dt, 1),
                               "кусков_с_пересинтезом": st.get("retry_chunks", 0),
                               "пересинтезов": st.get("retries", 0),
                               "пропало_или_искажено_по_судье": len(bad), "слова": bad}
        total_t += dt
        total_sec += len(audio) / SR
        agg["кусков"] += len(ch)
        agg["кусков_с_пересинтезом"] += st.get("retry_chunks", 0)
        agg["пересинтезов"] += st.get("retries", 0)
        print(name, json.dumps(res["тексты"][name], ensure_ascii=False), flush=True)
    res["итог"] = dict(agg, счёт_с=round(total_t, 1), звук_с=round(total_sec, 1),
                       RTF=round(total_t / max(total_sec, 1), 3),
                       пропало_всего=sum(v["пропало_или_искажено_по_судье"]
                                         for v in res["тексты"].values()))
    json.dump(res, open(os.path.join(a.out, "итог.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(res["итог"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
