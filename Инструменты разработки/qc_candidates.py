#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отсев ЯВНОГО брака перед слепым прослушиванием (не выбор победителя).

Для каждого варианта и текста:
  * сколько слов распознаётся обратно (Whisper large-v3-turbo) — выпавшие слова;
  * слышно ли первое и последнее слово — обрывы краёв;
  * срезанные вершины волны (клиппинг) и резкие скачки — треск, щелчки;
  * гейт voice_report против записи чтеца — только как справка.
Выбирает Андрей на слух: числа трижды расходились с ухом.

    python "Инструменты разработки/qc_candidates.py" --runs ПАПКА --codes A B32 … --inputs входы.json
"""
from __future__ import annotations

import argparse, difflib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = re.compile(r"[а-яё]+")


def words(s: str) -> list[str]:
    return W.findall(s.lower().replace("+", "").replace("ё", "е"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--codes", nargs="+", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import mlx_whisper
    import voice_report as vr
    items = {it["id"]: it for it in json.load(open(a.inputs, encoding="utf-8"))}
    longs = ["1_повествование", "2_диалог", "3_сложный"]
    ref = vr.resolve_source(json.load(open(os.path.expanduser(
        "~/.cache/text2audio/voices/Вячеслав Герасимов.src.json"), encoding="utf-8"))["source"])
    t = vr.analyze(ref)
    res = {}
    for code in a.codes:
        rd = os.path.join(a.runs, code)
        r = {"тексты": {}}
        for tid in longs + [k for k in items if k.startswith("фраза")]:
            p = os.path.join(rd, f"{tid}.wav")
            if not os.path.exists(p):
                r["тексты"][tid] = {"нет файла": True}
                continue
            y, sr = sf.read(p, dtype="float32")
            y = y.reshape(-1)
            want = words(items[tid]["plain"])
            got = words(mlx_whisper.transcribe(
                p, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language="ru", condition_on_previous_text=False)["text"])
            sm = difflib.SequenceMatcher(None, want, got, autojunk=False)
            match = sum(b.size for b in sm.get_matching_blocks()) / max(1, len(want))
            near = lambda w, among: any(difflib.SequenceMatcher(None, w, g).ratio() >= 0.7
                                        for g in among)
            pk = float(np.max(np.abs(y))) + 1e-9
            flat = np.abs(y) > 0.985 * pk
            tops = int(np.sum(flat[2:] & flat[1:-1] & flat[:-2]))
            d = np.abs(np.diff(y))
            clicks = int(np.sum(d > max(0.25, 40 * float(np.median(d)))))
            info = {"слов совпало, %": round(100 * match, 1),
                    "первое слово": bool(want) and near(want[0], got[:3]),
                    "последнее слово": bool(want) and near(want[-1], got[-3:]),
                    "срезанных вершин": tops, "резких скачков": clicks,
                    "распознано": " ".join(got)[:300]}
            if tid in longs:
                g = vr.analyze(p)
                info["гейт"] = {"металл": round(g["metal"], 3), "шипение": round(g["sibilance"], 3),
                                "треск/с": round(g["crackle"]), "F0": round(g["f0"], 1),
                                "воздух %": round(g["air"], 2)}
            r["тексты"][tid] = info
        lm = [r["тексты"][t_]["слов совпало, %"] for t_ in longs if "слов совпало, %" in r["тексты"].get(t_, {})]
        r["слов совпало на длинных, %"] = round(sum(lm) / max(1, len(lm)), 1)
        ph = [v for k, v in r["тексты"].items() if k.startswith("фраза") and "слов совпало, %" in v]
        r["фраз целиком распознано"] = sum(1 for v in ph if v["слов совпало, %"] >= 99.9)
        r["обрывов краёв"] = sum(1 for v in r["тексты"].values()
                                 if v.get("первое слово") is False or v.get("последнее слово") is False)
        res[code] = r
        print(code, json.dumps({k: v for k, v in r.items() if k != "тексты"}, ensure_ascii=False),
              flush=True)
    res["_эталон"] = {"металл": round(t["metal"], 3), "шипение": round(t["sibilance"], 3),
                      "треск/с": round(t["crackle"]), "F0": round(t["f0"], 1), "воздух %": round(t["air"], 2)}
    json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
