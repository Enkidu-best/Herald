#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Озвучить набор текстов вариантом НАШЕГО движка F5 (для слепого сравнения).

Варианты: веса (Misha accent_tune или ESpeech RL-V2), число шагов и расписание.
Каждый запуск — отдельный процесс, чтобы честно измерить пик памяти. Путь тот
же, что в программе: нарезка на куски, проверка краёв с переозвучкой.

    python "Инструменты разработки/run_f5_variant.py" --inputs входы.json \\
        --out ПАПКА --steps 33 --schedule uniform [--variant espeech]

«32 шага, как у F5»: в равномерном расписании N точек дают N-1 вызовов сети,
поэтому для 32 вызовов надо --steps 33 --schedule uniform.
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--schedule", default="epss")
    ap.add_argument("--variant", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import numpy as np
    import mlx.core as mx
    import soundfile as sf
    from core.engines import get_engine
    from core.pipeline import _chunk_paragraphs, _synth_chunks

    kw = dict(language="ru", steps=a.steps, schedule=a.schedule)
    if a.variant:
        kw["variant"] = a.variant
    t0 = time.time()
    eng = get_engine("f5mlx", **kw)
    eng.ensure_loaded()
    meas = {"вариант": kw, "загрузка, с": round(time.time() - t0, 1), "тексты": {}}
    sr = eng.sample_rate
    tot_a = tot_t = 0.0
    for it in json.load(open(a.inputs, encoding="utf-8")):
        chunks = _chunk_paragraphs(it["stressed"], eng.max_chunk_chars, 0.35, 0.6)
        t1 = time.time()
        audio = _synth_chunks(eng, a.voice, chunks, sr, 0.35, lang="ru")
        took = time.time() - t1
        sf.write(os.path.join(a.out, f"{it['id']}.wav"), audio, sr)
        dur = len(audio) / sr
        tot_a += dur
        tot_t += took
        meas["тексты"][it["id"]] = {"звук, с": round(dur, 1), "счёт, с": round(took, 1)}
        print(f"  {it['id']}: {dur:.1f} c звука за {took:.1f} c", flush=True)
    meas.update({"звук всего, с": round(tot_a, 1), "счёт всего, с": round(tot_t, 1),
                 "с счёта на с звука": round(tot_t / max(tot_a, 1e-6), 3),
                 "пик памяти MLX, ГБ": round(mx.get_peak_memory() / 1e9, 2)})
    json.dump(meas, open(os.path.join(a.out, "замер.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meas.items() if k != "тексты"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
