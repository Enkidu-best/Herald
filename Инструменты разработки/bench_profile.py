#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Куда уходит время озвучки (3.5, п. 1): по шагам, как в окне.

Книга озвучивается обычным convert_book (проверка слов по звуку включена,
кэш выключен), а на шаги навешены часы: подготовка текста, проход модели
(решение ОДУ), вокодер и хвост, распознавание краёв (Whisper), проверка слов
по звуку (выравнивание), доводка пауз, сборка mp3. Синтез и проверки идут
параллельно (упреждение), поэтому сумма шагов больше общего времени.

    python "Инструменты разработки/bench_profile.py" --out профиль.json [--max-chars 280]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import collections
import json
import tempfile
import threading
import time

T = collections.defaultdict(float)
N = collections.Counter()
LOCK = threading.Lock()


def timed(obj, name, key):
    f = getattr(obj, name)

    def w(*a, **k):
        t0 = time.monotonic()
        try:
            return f(*a, **k)
        finally:
            with LOCK:
                T[key] += time.monotonic() - t0
                N[key] += 1
    setattr(obj, name, w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="Примеры текстов/Пример - проба пера.txt")
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-edges", action="store_true", help="без распознавания краёв (граница выигрыша)")
    ap.add_argument("--set", action="append", default=[], metavar="МОДУЛЬ.ИМЯ=ЗНАЧЕНИЕ",
                    help="подменить константу на время замера (варианты п. 1)")
    ap.add_argument("--kw", action="append", default=[], metavar="ИМЯ=ЧИСЛО",
                    help="параметр движка (например cfg_steps=8)")
    ap.add_argument("--cold", action="store_true",
                    help="как сразу после открытия окна: модели не загружены, загрузка - в счёт")
    ap.add_argument("--quality", default=None, help="режим fast / balanced / max")
    ap.add_argument("--engine", default=None, help="silero - готовый голос вместо клона")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import core.pipeline as P
    from core import profiles, assemble
    from core.engines import clone_engine
    from core.engines import f5_mlx as F
    for s in a.set:
        k, v = s.split("=", 1)
        mod, name = k.rsplit(".", 1)
        import importlib
        m = importlib.import_module(mod)
        setattr(m, name, type(getattr(m, name))(v) if not isinstance(getattr(m, name), bool)
                else v.lower() in ("1", "true", "да"))
        print("подмена:", k, "=", getattr(m, name))
    kw = (profiles.engine_kwargs(profiles.get(None), a.quality) if a.quality
          else dict(profiles.get(None).kwargs))
    for s_ in a.kw:
        k_, v_ = s_.split("=", 1)
        kw[k_] = int(v_) if v_.lstrip("-").isdigit() else v_
    engine_id = a.engine or clone_engine()
    if a.engine == "silero":
        kw = {}
    from core import align
    if a.cold:
        # загрузка моделей - в счёт, как в окне сразу после открытия
        timed(F.F5MLXEngine, "_load_in_thread", "загрузка клона F5")
        timed(align.Aligner, "load", "загрузка выравнивателя (проверка слов)")
        from core import asr
        timed(asr, "transcribe", "_распознавание (с загрузкой Whisper)")
        timed(P, "_loaded_engine", "_взять движок (с загрузкой)")
    else:
        eng = P._loaded_engine(engine_id, dict(kw, language="ru") if a.engine != "silero" else {})
        if a.max_chars:
            type(eng).max_chunk_chars = a.max_chars
        eng.synth_batch(["Пров+ерка св+язи."], a.voice)             # прогрев
        if not a.no_check:
            align.aligner().load()
        P._listen(eng.synth_batch(["Пров+ерка."], a.voice)[0], 24000, "ru", P._ASR_MODEL)
    if a.no_edges:
        P._listen = lambda *x, **k: None
    timed(P, "_prepare_text", "подготовка текста (числа, ударения)")
    timed(F.F5MLXEngine, "_sample_mel", "модель: решение ОДУ (12 шагов)")
    timed(F.F5MLXEngine, "_synth_in_thread", "_синтез в потоке MLX (модель+вокодер)")
    timed(P, "_listen", "проверка краёв (Whisper)")
    timed(P, "_sound_words", "проверка слов по звуку (выравнивание)")
    timed(P, "_fit_inner_pauses", "доводка пауз внутри куска")
    timed(P, "_synth_by_sentences", "_чтение куска по предложениям (пересчёт)")
    timed(assemble, "encode_chapter_mp3", "сборка mp3 (ffmpeg)")
    calls = []
    orig = F.F5MLXEngine.synth_batch

    def count(self, texts, *x, **k):
        calls.append(len(texts))
        return orig(self, texts, *x, **k)
    F.F5MLXEngine.synth_batch = count
    timed(F.F5MLXEngine, "synth_batch", "_синтез целиком (вызовы движка)")
    out = tempfile.mkdtemp(prefix="herald_prof_")
    opts = P.ConvertOptions(use_cache=False, sound_check=not a.no_check)
    stats = {}
    t0 = time.monotonic()
    res = P.convert_book(a.book, engine=engine_id, eng_kwargs=dict(kw), voice=a.voice,
                         out_dir=out, options=opts, on_rtf=lambda r: stats.setdefault("rtf", r))
    wall = time.monotonic() - t0
    import soundfile as sf
    mp3s = [_os.path.join(dp, f) for dp, _d, fs in _os.walk(out) for f in fs if f.endswith(".mp3")]
    dur = 0.0
    import subprocess
    for m in mp3s:
        dur += float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", m], capture_output=True, text=True).stdout)
    T["_вокодер и хвост (синтез - ОДУ)"] = T["_синтез в потоке MLX (модель+вокодер)"] - T["модель: решение ОДУ (12 шагов)"]
    print(f"\nзвук {dur:.1f} с, всего {wall:.1f} с, RTF {wall / dur:.2f}; вызовов движка "
          f"{len(calls)}, кусков в них {sum(calls)}")
    for k in sorted(T, key=lambda k: -T[k]):
        print(f"  {k:48s} {T[k]:7.1f} с  ({T[k] / dur:.2f} на с звука, вызовов {N.get(k, '-')})")
    if a.out:
        json.dump({"звук_с": dur, "всего_с": wall, "rtf": wall / dur, "шаги": dict(T),
                   "вызовов": dict(N), "кусков_в_вызовах": sum(calls)},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
