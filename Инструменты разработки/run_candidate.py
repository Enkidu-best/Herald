#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Озвучить набор текстов кандидатом из mlx-audio (Qwen3-TTS, Chatterbox).

Запускается в ОТДЕЛЬНОМ окружении .venv-candidates: mlx-audio тянет свои
версии mlx и transformers, и ставить его в основное .venv значит рисковать
одобренным движком. Основное окружение этот скрипт не трогает.

    .venv-candidates/bin/python "Инструменты разработки/run_candidate.py" \\
        --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --inputs входы.json \\
        --ref образец.wav --ref-text "…" --out ПАПКА

На выходе: по WAV на каждый текст и замер.json (время, длина звука, память).
Длинные тексты режутся на куски по предложениям (до ~250 знаков) — так же, как
режет наш конвейер: у моделей-«языковых» на длинном тексте бывают сбои, и
сравнивать их надо в тех же условиях, в каких они работали бы в программе.
"""
from __future__ import annotations

import argparse, json, os, re, time
import numpy as np


def chunks(text: str, limit: int = 250) -> list[str]:
    sents = [s for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--key", default="plain", help="какое поле входа озвучивать")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import mlx.core as mx
    import soundfile as sf
    from mlx_audio.tts.utils import load_model

    t0 = time.time()
    model = load_model(a.model)
    load_s = time.time() - t0
    sr = getattr(model, "sample_rate", 24000)
    items = json.load(open(a.inputs, encoding="utf-8"))
    meas = {"model": a.model, "загрузка, с": round(load_s, 1), "тексты": {}}
    is_qwen = "qwen" in a.model.lower()
    total_audio = total_time = 0.0
    for it in items:
        text = it[a.key]
        pieces = []
        t1 = time.time()
        for ch in chunks(text):
            kw = dict(text=ch, ref_audio=a.ref, lang_code=a.lang, verbose=False)
            if is_qwen:
                kw.update(ref_text=a.ref_text, lang_code="russian")
            try:
                for res in model.generate(**kw):
                    pieces.append(np.asarray(res.audio, dtype=np.float32).reshape(-1))
                    sr = getattr(res, "sample_rate", sr) or sr
            except Exception as e:
                print(f"  ! {it['id']}: кусок не озвучен: {e}", flush=True)
            pieces.append(np.zeros(int(0.35 * sr), dtype=np.float32))
        took = time.time() - t1
        audio = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
        sf.write(os.path.join(a.out, f"{it['id']}.wav"), audio, sr)
        dur = len(audio) / sr
        total_audio += dur
        total_time += took
        meas["тексты"][it["id"]] = {"звук, с": round(dur, 1), "счёт, с": round(took, 1)}
        print(f"  {it['id']}: {dur:.1f} c звука за {took:.1f} c", flush=True)
    meas["звук всего, с"] = round(total_audio, 1)
    meas["счёт всего, с"] = round(total_time, 1)
    meas["с счёта на с звука"] = round(total_time / max(total_audio, 1e-6), 3)
    meas["пик памяти MLX, ГБ"] = round(mx.get_peak_memory() / 1e9, 2)
    json.dump(meas, open(os.path.join(a.out, "замер.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meas.items() if k != "тексты"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
