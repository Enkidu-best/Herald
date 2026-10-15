#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-движок (Windows, Linux) считает то же, что MLX-движок на Mac (3.2).

Один и тот же кусок, один и тот же начальный шум, 12 шагов EPSS, якорь тишины:
мел-спектр torch и MLX должны совпасть с точностью float32 (сеть та же, веса
те же, алгоритм общий - отличается только библиотека счёта). Потом - звук
целиком через torch: длительность, хвост, распознавание.

    python "Инструменты разработки/test_torch_parity.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import time

import numpy as np

VOICE = "Вячеслав Герасимов"
TEXT = "Мор+е к в+ечеру ст+ихло, и над вод+ой пов+ис тум+ан."
fails: list[str] = []


def must(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def main() -> int:
    import mlx.core as mx
    import core.engines.f5_mlx as F
    from core.engines.f5_mlx import F5MLXEngine
    from core.engines.f5_torch import F5TorchEngine
    import core.pipeline as P

    em = F5MLXEngine(language="ru", steps=12, schedule="epss")
    et = F5TorchEngine(language="ru", steps=12, schedule="epss", device="cpu")
    em.ensure_loaded()
    t0 = time.time()
    et.ensure_loaded()
    print(f"  torch загружен за {time.time() - t0:.0f} с")
    audio_m, rt, _g, em._pace = em._get_ref(VOICE)
    audio_t, rt2, _g2, et._pace = et._get_ref(VOICE)
    must(rt == rt2 and abs(em._pace - et._pace) < 1e-6, "образец и темп одинаковы")
    ref_frames = audio_t.shape[0] // F.HOP_LENGTH
    g = P._with_end_punct(TEXT)
    dur = em._gen_frames(ref_frames, rt, g, 1.0, 1.0)
    must(dur == et._gen_frames(ref_frames, rt, g, 1.0, 1.0), "длительность одинакова")
    tokens = F.convert_char_to_tokens([rt + g])[0]
    tail = int(F.TAIL_ANCHOR_SEC * F.SAMPLE_RATE / F.HOP_LENGTH)
    n = max(ref_frames + 1, dur, len(tokens)) + tail
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((n, 100)).astype(np.float32)

    def mlx_run():
        mel, lens = em._sample_mel(audio_m, [tokens], [dur], [mx.array(noise)], tails=[tail])
        return np.array(mel[0, :lens[0]])
    a = F._MLX_POOL.submit(mlx_run).result()
    t0 = time.time()
    b, nb = F._MLX_POOL.submit(et._sample_mel_torch, audio_t, tokens, dur, tail, noise).result()
    b = b[0, :nb].numpy()
    print(f"  torch (процессор): {time.time() - t0:.1f} с на кусок")
    must(a.shape == b.shape, f"размер мела {a.shape} = {b.shape}")
    gen = slice(ref_frames, n - tail)
    diff = float(np.max(np.abs(a[gen] - b[gen]))) if a.shape == b.shape else 1e9
    rel = diff / float(np.max(np.abs(a[gen])) + 1e-9)
    must(rel < 0.02, f"мел torch = MLX: наибольшее расхождение {diff:.3g} ({rel:.2%} от размаха)")

    print("Звук через torch целиком:")
    w = et.synth_chunk(g, VOICE)
    sec = len(w) / F.SAMPLE_RATE
    from core.vad import speech_end
    se = speech_end(w, F.SAMPLE_RATE)
    must(2.0 < sec < 10.0, f"длительность {sec:.1f} с")
    must(se is not None and sec - se >= 0.2 - 1e-3, f"после речи {sec - (se or 0):.2f} с хвоста")
    heard = P._listen(w, F.SAMPLE_RATE, "ru", "turbo")
    bad = P._bad_words(heard, g)
    must(P._edges_verdict(heard, g, sec) and not bad,
         f"распознано: {' '.join(x for x, _s, _e in heard or [])} (не так: {bad})")
    print("\nOK: torch = MLX" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
