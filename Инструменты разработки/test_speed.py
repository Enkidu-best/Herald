#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скорость (Herald 3.1, раздел 3): профили, пачки, кэш синтеза.

  * профили: основной и ESpeech - 12 вызовов сети (3.2, 3.3), черновик - 7;
    32 шагов в окне нет;
  * пачка = по одному: два куска разной длины одним проходом дают тот же мел,
    что и по отдельности (тот же начальный шум; добивка закрыта маской);
  * кэш: ключ меняется от текста, темпа, движка; повторная озвучка того же
    куска берётся из кэша и совпадает с первой; кусок с недоговорённым концом
    в кэш не попадает;
  * упреждение: пачки с упреждением и без дают одинаковую последовательность
    кусков (порядок и паузы).
Сам RTF «до/после» меряет стенд bench_speed.py (это минуты счёта).

    python "Инструменты разработки/test_speed.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import tempfile

import numpy as np

VOICE = "Вячеслав Герасимов"
fails: list[str] = []


def must(ok: bool, msg: str):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        fails.append(msg)


def main() -> int:
    import mlx.core as mx
    from core import profiles, synthcache
    from core.engines import get_engine
    import core.engines.f5_mlx as F

    print("Профили:")
    want = {"main": 12, "espeech": 12, "draft": 7}
    for key, n in want.items():
        p = profiles.PROFILES[key]
        e = get_engine("f5mlx", language="ru", **p.kwargs)
        calls = e._timesteps().shape[0] - 1
        must(calls == n, f"«{p.title}»: вызовов сети {calls}")
    must(profiles.DEFAULT_PROFILE == "main" and "variant" not in profiles.get(None).kwargs,
         "по умолчанию - Misha на 12 шагах (выбор Андрея 15.10)")
    must({k for k, p in profiles.PROFILES.items() if p.engine.startswith("f5")} == set(want),
         "32 шага в окно не выведены")
    must(not any(ch.isdigit() for p in profiles.PROFILES.values() for ch in p.title),
         "в названиях профилей нет цифр")

    print("Пачка = по одному:")
    eng = get_engine("f5mlx", language="ru", steps=7, schedule="epss")
    eng.ensure_loaded()
    audio, ref_text, _g, eng._pace = eng._get_ref(VOICE)
    ref_frames = audio.shape[0] // F.HOP_LENGTH
    texts = ["Дом сто+ял на краю дер+евни.",
             "К+аждое +утро над ним подним+ался д+ым, и с+оседи зн+али: хоз+яин д+ома."]
    toks = F.convert_char_to_tokens([ref_text + t for t in texts])
    durs = [eng._gen_frames(ref_frames, ref_text, t, 1.0, 1.0) for t in texts]
    mx.random.seed(0)
    # всё MLX - в его общем потоке (иначе «There is no Stream(gpu, 0)»)
    noise = F._MLX_POOL.submit(lambda: [mx.random.normal((d, 100)) for d in durs]).result()

    def run(tk, ds, nz):
        return F._MLX_POOL.submit(eng._sample_mel, audio, tk, ds, nz).result()
    both, lens = run(toks, durs, noise)
    for i in range(2):
        one, l1 = run([toks[i]], [durs[i]], [noise[i]])
        a = np.array(both[i, :lens[i]])
        b = np.array(one[0, :l1[0]])
        diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
        must(lens[i] == l1[0] and diff < 2e-3,
             f"кусок {i + 1}: длина {lens[i]} = {l1[0]}, расхождение мела {diff:.1e}")
    w = eng.synth_batch(texts, VOICE)
    must(len(w) == 2 and all(len(x) > 24000 * 0.8 for x in w), "synth_batch дал оба куска")

    print("Кэш синтеза:")
    old_dir = synthcache.CACHE_DIR
    synthcache.CACHE_DIR = tempfile.mkdtemp(prefix="herald_cache_test_")
    try:
        vd = F.F5Engine.__init__.__globals__["default_voices_dir"]()
        k1 = synthcache.key(eng.cache_key(), vd, VOICE, texts[0], 1.0)
        must(k1 == synthcache.key(eng.cache_key(), vd, VOICE, texts[0], 1.0), "ключ устойчив")
        must(k1 != synthcache.key(eng.cache_key(), vd, VOICE, texts[0] + " ", 1.0),
             "другой текст - другой ключ")
        must(k1 != synthcache.key(eng.cache_key(), vd, VOICE, texts[0], 0.94),
             "другой темп - другой ключ")
        e16 = get_engine("f5mlx", language="ru", steps=16, schedule="epss")
        must(k1 != synthcache.key(e16.cache_key(), vd, VOICE, texts[0], 1.0),
             "другое качество - другой ключ")
        x = (np.sin(np.arange(24000) / 7) * 0.3).astype(np.float32)
        synthcache.put(k1, x, 24000, [("дом", 0.1, 0.3)])
        got = synthcache.get(k1)
        must(got is not None and np.max(np.abs(got[0] - x)) < 1e-5 and got[1] == [("дом", 0.1, 0.3)],
             "запись и чтение без потерь (FLAC 24 бит + слова)")

        from core.pipeline import _synth_chunks
        chunk = [("Дом сто+ял на краю дер+евни, у с+амой реки.", 0.4, 1.0, ())]
        st1, st2 = {}, {}
        a1 = _synth_chunks(eng, VOICE, chunk, 24000, 0.4, stats=st1)
        a2 = _synth_chunks(eng, VOICE, chunk, 24000, 0.4, stats=st2)
        must(st1["cached"] == 0 and st2["cached"] == 1, f"второй раз из кэша ({st2})")
        must(len(a1) == len(a2) and float(np.max(np.abs(a1 - a2))) < 1e-4,
             "из кэша - тот же звук")
        must(st2["fresh_sec"] == 0, "из кэша ничего не считалось заново")

        print("Упреждение:")
        many = [(f"Фр+аза н+омер {i}, просто для пров+ерки пор+ядка.", 0.4, 1.0, ())
                for i in range(1, 6)]
        _synth_chunks(eng, VOICE, many, 24000, 0.4, batch_size=2)   # всё в кэш
        b1 = _synth_chunks(eng, VOICE, many, 24000, 0.4, batch_size=2, pipelined=True)
        b2 = _synth_chunks(eng, VOICE, many, 24000, 0.4, batch_size=3, pipelined=False)
        must(len(b1) == len(b2) and float(np.max(np.abs(b1 - b2))) < 1e-4,
             "порядок и паузы те же при любом размере пачки и упреждении")
    finally:
        import shutil
        shutil.rmtree(synthcache.CACHE_DIR, ignore_errors=True)
        synthcache.CACHE_DIR = old_dir

    if fails:
        print(f"\nПРОВАЛ: {len(fails)}")
        return 1
    print("\nOK: скорость")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
