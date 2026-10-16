#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка краёв (3.5): Whisper small (как было) против выравнивания (core/align).

На наборе окончаний 3.1 (окончания_33.json: три места Андрея и 30 фраз на
глухие концы) - сырые дубли модели, по нескольку зёрен. Каждый дубль судят:
  * Whisper large-v3-turbo - судья стенда окончаний (последние буквы);
  * прежняя проверка краёв программы (Whisper small, _edges_verdict);
  * новая - по звуку (align.edges_ok).
Ни один судья не истина (Whisper дописывает по смыслу), поэтому печатается
согласие и места Андрея отдельно.

    python "Инструменты разработки/eval_edges.py" --seeds 3 --out края.json
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import contextlib
import io
import json
import tempfile

SR = 24000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import mlx.core as mx
    import mlx_whisper
    import soundfile as sf
    import core.pipeline as P
    from core import profiles, align
    from core.engines import get_engine, clone_engine
    from core.vad import speech_end
    sets = json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                        "окончания_33.json"), encoding="utf-8"))
    eng = get_engine(clone_engine(), language="ru", **profiles.get(None).kwargs)
    eng.ensure_loaded()
    al = align.aligner().load()

    def words(s):
        import re
        return re.sub(r"[^\w ]", " ", s.lower().replace("+", "").replace("ё", "е")).split()

    def turbo(audio):
        se = speech_end(audio, SR)
        if se is not None:
            audio = audio[:int((se + 0.2) * SR)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, SR)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = mlx_whisper.transcribe(f.name, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                           language="ru", verbose=False, word_timestamps=True,
                                           condition_on_previous_text=False)
        finally:
            _os.unlink(f.name)
        out = []
        for seg in r["segments"]:
            for w in seg.get("words", []):
                if se is not None and w["start"] > se + 0.05:
                    continue
                out += words(w["word"])
        return out

    rows = []
    for group, phrases in sets.items():
        for ph in phrases:
            text = P._with_end_punct(ph if "+" in ph else P._prepare_text(ph, "+", True, True, "ru"))
            want = words(text)
            for seed in range(a.seeds):
                mx.random.seed(1000 + seed)
                audio = eng.synth_batch([text], a.voice)[0]
                h = turbo(audio)
                k = 3 if len(want[-1]) >= 5 else 2
                t_ok = bool(h) and (h[-1] == want[-1] or h[-1][-k:] == want[-1][-k:])
                old_ok = P._edges_verdict(P._listen(audio, SR, "ru", P._ASR_MODEL), text,
                                          len(audio) / SR)
                ws = al.score(audio, SR, text.replace("+", ""))
                new_ok = align.edges_ok(ws, text)
                rows.append({"набор": group, "фраза": ph[:60], "зерно": seed, "turbo": t_ok,
                             "прежняя": old_ok, "новая": new_ok,
                             "turbo_слышит": " ".join(h[-2:]),
                             "модель_слышит": ws[-1].heard if ws else ""})
                print(f"{group:6s} {seed} turbo {'да' if t_ok else 'НЕТ'} | прежняя "
                      f"{'да' if old_ok else 'НЕТ'} | новая {'да' if new_ok else 'НЕТ'} | "
                      f"«{rows[-1]['turbo_слышит']}» / «{rows[-1]['модель_слышит']}» | {ph[:40]}",
                      flush=True)
    n = len(rows)
    for name in ("прежняя", "новая"):
        agree = sum(r[name] == r["turbo"] for r in rows)
        caught = sum(1 for r in rows if not r["turbo"] and not r[name])
        bad = sum(1 for r in rows if not r["turbo"])
        false = sum(1 for r in rows if r["turbo"] and not r[name])
        print(f"{name}: согласие с turbo {agree}/{n}; turbo-брак пойман {caught}/{bad}; "
              f"бракует, где turbo доволен, {false}/{n - bad}")
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
