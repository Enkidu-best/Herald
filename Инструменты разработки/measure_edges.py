#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер «было -> стало» для краёв кусков: съеденное последнее слово и щелчок.

Один и тот же набор кусков (30 фраз + повествование + диалог) синтезируется в
старом и новом режиме с ОДНИМ И ТЕМ ЖЕ зерном случайности на кусок - поэтому
разница идёт от правок, а не от случайного шума модели.

  * неполное последнее слово - по Whisper large-v3-turbo (сильнее, чем small,
    которым пользуется сама проверка краёв, - чтобы оценка была независимой);
  * щелчок - резкие скачки сигнала в последних 100 мс вместе с переходом в паузу.

    python "Инструменты разработки/measure_edges.py" --inputs входы.json \\
        --mode new --variant misha --out ПАПКА
"""
from __future__ import annotations

import argparse, difflib, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = re.compile(r"[а-яё]+")


def words(s):
    return W.findall(s.lower().replace("+", "").replace("ё", "е"))


def complete(want, heard):
    for h in heard[-2:]:
        if h == want or (difflib.SequenceMatcher(None, want, h).ratio() >= 0.75
                         and h[-2:] == want[-2:]):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--variant", choices=["misha", "espeech"], default="misha")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import numpy as np, soundfile as sf, mlx.core as mx, mlx_whisper
    import core.engines.f5_mlx as F
    import core.pipeline as P
    old = a.mode == "old"
    F.TAIL_MODE = "rms" if old else "vad"
    P.STRICT_EDGES = not old
    P.END_PUNCT = not old
    from core.engines import get_engine
    kw = dict(language="ru", steps=16, schedule="epss")
    if a.variant == "espeech":
        kw["variant"] = "espeech"
    eng = get_engine("f5mlx", **kw)
    eng.ensure_loaded()
    sr, gap = eng.sample_rate, 0.35
    items = [it for it in json.load(open(a.inputs, encoding="utf-8"))
             if it["id"].startswith("фраза") or it["id"] in ("1_повествование", "2_диалог")]
    chunks = []
    for it in items:
        for c in P._chunk_paragraphs(it["stressed"], eng.max_chunk_chars, gap, 0.6):
            chunks.append((it["id"], c))
    res, t0 = [], time.time()
    for i, (iid, ch) in enumerate(chunks):
        mx.random.seed(1000 + i)
        out = P._synth_chunks(eng, "Вячеслав Герасимов", [ch], sr, gap, lang="ru")
        audio = out[:-int(sr * gap)] if len(out) > int(sr * gap) else out
        p = os.path.join(a.out, f"{i:03d}.wav")
        sf.write(p, audio, sr)
        want = words(ch[0])
        tail = audio[-int(min(len(audio) / sr, 4.0) * sr):]
        sf.write(p + ".tail.wav", tail, sr)
        got = words(mlx_whisper.transcribe(p + ".tail.wav",
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    language="ru", condition_on_previous_text=False)["text"])
        os.remove(p + ".tail.wav")
        x = np.concatenate([audio[-int(0.1 * sr):], np.zeros(int(0.02 * sr), np.float32)])
        jumps = int(np.sum(np.abs(np.diff(x)) > 0.05))
        res.append({"id": iid, "кусок": i, "последнее": want[-1] if want else "",
                    "услышано": " ".join(got[-3:]),
                    "полное": bool(want) and complete(want[-1], got),
                    "скачков в конце": jumps, "последний отсчёт": round(float(abs(audio[-1])), 4)})
    summ = {"режим": a.mode, "вариант": a.variant, "кусков": len(res),
            "неполное последнее слово": sum(1 for r in res if not r["полное"]),
            "кусков со щелчком в конце": sum(1 for r in res if r["скачков в конце"] > 0),
            "скачков в конце всего": sum(r["скачков в конце"] for r in res),
            "время, с": round(time.time() - t0)}
    json.dump({"итог": summ, "куски": res}, open(os.path.join(a.out, "итог.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
