#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Окончания на концах кусков: замер «было -> стало» (Herald 3.1, п. 2.4).

Набор - окончания_33.json: три места, где Андрей слышал потери, и 30 фраз на
глухие концы (-г, -к, -т, -ть, -сть, -кт), каждая - отдельным куском, то есть
последнее слово стоит в самом конце куска.

Код можно взять другой (--code-root): так «было» считается настоящим кодом 3.0
из снимка, а не его имитацией. На каждую фразу и зерно:
  * сырой выход модели (до обработки) - есть ли ещё звук у самой границы окна
    («край», дБ к громкости речи: выше -30 - слово упёрлось в границу);
  * итог конвейера (с его переозвучками) - так кусок попадёт в книгу:
    Whisper large-v3-turbo (не тот, что в программе), последнее слово должно
    совпасть с текстом КОНЦОМ - последними тремя буквами (двумя у коротких).

    python "Инструменты разработки/bench_endings.py" --out ПАПКА [--code-root СНИМОК] [--seeds 2]
"""
from __future__ import annotations

import argparse, contextlib, inspect, io, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-root", default=os.path.dirname(HERE))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.code_root))
    os.makedirs(a.out, exist_ok=True)
    import numpy as np, soundfile as sf, mlx.core as mx, mlx_whisper
    import core.engines.f5_mlx as F
    import core.pipeline as P
    from core.engines import get_engine
    print("код:", P.__file__, flush=True)

    hook = "_finish_tail" if hasattr(F, "_finish_tail") else "_trim_tail"
    orig = getattr(F, hook)
    raws = []

    def spy(w):
        raws.append(np.array(w, copy=True))
        return orig(w)
    setattr(F, hook, spy)
    anchor = getattr(F, "TAIL_ANCHOR_SEC", 0.0)

    eng = get_engine("f5mlx", language="ru", steps=a.steps, schedule="epss")
    eng.ensure_loaded()
    kw = {"use_cache": False} if "use_cache" in inspect.signature(P._synth_chunks).parameters else {}

    def words(s):
        import re
        return re.sub(r"[^\w ]", " ", s.lower().replace("+", "").replace("ё", "е")).split()

    from core.vad import speech_end

    def judge(audio):
        """Слова итога; слова, начатые уже в тишине после речи, - выдумка Whisper."""
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
            os.unlink(f.name)
        out = []
        for seg in r["segments"]:
            for w in seg.get("words", []):
                if se is not None and w["start"] > se + 0.05:
                    continue
                out += words(w["word"])
        return out

    def end_ok(want, heard):
        if not heard:
            return False
        h = heard[-1]
        k = 3 if len(want) >= 5 else 2
        return h == want or h[-k:] == want[-k:]

    def edge_db(raw):
        w = 240
        end = len(raw) - int(anchor * SR)       # граница, где кончается свобода модели
        seg = raw[:end]
        n = len(seg) // w
        e = np.sqrt(np.mean(seg[:n * w].reshape(n, w) ** 2, axis=1))
        ref = float(np.percentile(e, 95)) + 1e-9
        return float(20 * np.log10(e[-3:-1].mean() / ref + 1e-9))

    data = json.load(open(os.path.join(HERE, "окончания_33.json"), encoding="utf-8"))
    rows = []
    for group, phrases in data.items():
        for pi, text in enumerate(phrases):
            stressed = text if "+" in text else P._prepare_text(text, "+", True, True, "ru")
            want = words(text)[-1]
            for seed in range(a.seeds):
                mx.random.seed(500 + 17 * pi + seed)
                raws.clear()
                final = P._synth_chunks(eng, a.voice, [(stressed, 0.4, 1.0)], SR, 0.4, **kw)
                raw = raws[0]                       # первая попытка, до переозвучек
                heard = judge(final)
                rec = {"группа": group, "фраза": text, "зерно": seed, "слово": want,
                       "услышано": " ".join(heard[-2:]), "конец_цел": end_ok(want, heard),
                       "край_дБ": round(edge_db(raw), 1), "попыток": len(raws)}
                rows.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                if group == "Андрей":
                    sf.write(os.path.join(a.out, f"Андрей_{pi + 1}_{seed}.wav"), final, SR)
    setattr(F, hook, orig)
    n = len(rows)
    ok = sum(r["конец_цел"] for r in rows)
    cut = sum(r["край_дБ"] > -30 for r in rows)
    summ = {"код": P.__file__, "дублей": n, "последнее слово целиком": ok,
            "слово упёрлось в край окна (сырой выход)": cut,
            "переозвучек": sum(r["попыток"] - 1 for r in rows),
            "по группам": {g: f"{sum(r['конец_цел'] for r in rows if r['группа'] == g)}/"
                              f"{sum(1 for r in rows if r['группа'] == g)}" for g in data}}
    json.dump({"итог": summ, "дубли": rows}, open(os.path.join(a.out, "итог.json"), "w",
                                                  encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
