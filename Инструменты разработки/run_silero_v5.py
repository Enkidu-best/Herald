#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Silero v5 ru — ориентир «чистого» звука для слепого сравнения (не клон).

Готовый голос диктора Silero, не наш чтец. Нужен, чтобы услышать, как звучит
синтез без «металла» вокодера F5. Лицензия Silero — CC BY-NC 4.0 (только
некоммерческое использование). Ударения — наши (RUAccent + омографы от
silero-stress), в том же виде, что получают кандидаты F5.

    python "Инструменты разработки/run_silero_v5.py" --inputs входы.json --out ПАПКА
"""
from __future__ import annotations

import argparse, json, os, re, resource, time

MODEL = os.path.expanduser("~/.cache/text2audio/silero/v5_5_ru.pt")


def chunks(text: str, limit: int = 800) -> list[str]:
    sents = [s for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    return out + ([cur] if cur else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", default="aidar")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    model = torch.package.PackageImporter(MODEL).load_pickle("tts_models", "model")
    meas = {"модель": "silero v5_5_ru", "голос": a.speaker,
            "загрузка, с": round(time.time() - t0, 1), "тексты": {}}
    sr = 48000
    ta = tt = 0.0
    for it in json.load(open(a.inputs, encoding="utf-8")):
        t1 = time.time()
        pieces = []
        for ch in chunks(it["stressed"]):
            try:
                pieces.append(model.apply_tts(text=ch, speaker=a.speaker, sample_rate=sr,
                                              put_accent=True, put_yo=True).numpy())
            except Exception as e:
                print(f"  ! {it['id']}: {e}", flush=True)
            pieces.append(np.zeros(int(0.35 * sr), dtype=np.float32))
        took = time.time() - t1
        audio = np.concatenate(pieces).astype(np.float32)
        sf.write(os.path.join(a.out, f"{it['id']}.wav"), audio, sr)
        d = len(audio) / sr
        ta += d
        tt += took
        meas["тексты"][it["id"]] = {"звук, с": round(d, 1), "счёт, с": round(took, 1)}
        print(f"  {it['id']}: {d:.1f} c за {took:.1f} c", flush=True)
    meas.update({"звук всего, с": round(ta, 1), "счёт всего, с": round(tt, 1),
                 "с счёта на с звука": round(tt / max(ta, 1e-6), 3),
                 "пик памяти процесса, ГБ": round(
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)})
    json.dump(meas, open(os.path.join(a.out, "замер.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meas.items() if k != "тексты"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
