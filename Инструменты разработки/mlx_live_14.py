#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 14 В, живая проверка - под настоящим замком видеокарты (фон), своя папка данных (стандарт 6.10):

  1. три озвучки подряд (F5-MLX) - без «no Stream»;
  2. озвучка после искусственного сбоя в нити MLX;
  3. проверка слов по звуку (mlx_whisper) из другой нити во время озвучки («Прочитать вслух» во время книги);
  4. замер «На процессоре» для окна «Видеокарта занята» (доп. 21:40): тот же текст через F5 на torch на процессоре -
     во сколько раз медленнее MLX на этом Mac. Итог - «bench_out/mlx_live_14.json» (не в git).

    HEAVY_LOCK_WHO=background .venv/bin/python "Инструменты разработки/mlx_live_14.py" [--no-cpu]

Книгу Андрея, его голоса и настройки не трогает (временная папка; модели - общие, только чтение).
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
REAL_LOCK = HERE.parent.parent / ".heavy-lock"          # настоящая очередь «Разработка ПО»
import herald_testdata as TD  # noqa: E402

TD.setup(book=False)
os.environ["HERALD_HEAVY_LOCK"] = str(REAL_LOCK)       # прогон с моделью - под настоящим замком
from common.lock import heavy_lock as hl  # noqa: E402

TEXT = ("Море к вечеру стихло, и над водой повис туман. Старый смотритель поднялся наверх и зажёг огонь, "
        "чтобы корабли нашли дорогу к берегу.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cpu", action="store_true")
    a = ap.parse_args()
    lock = hl.HeavyLock("Herald (проверка 14 В)", "три озвучки, сбой, проверка слов, замер процессора",
                        who=os.environ.get("HEAVY_LOCK_WHO") or "background", path=REAL_LOCK, idle_after=None)
    if not lock.acquire(poll=10, on_wait=lambda h, n: print(f"жду видеокарту: {h.program if h else 'очередь'}",
                                                            flush=True)):
        return 75
    res: dict = {"начало": time.strftime("%H:%M:%S")}
    try:
        import numpy as np
        import soundfile as sf
        from core import asr, engines, mlxthread
        from core.engines.f5_mlx import SAMPLE_RATE as SR, _in_mlx_thread
        eng = engines.get_engine("f5mlx")
        t0 = time.time()
        eng.load()
        res["загрузка F5-MLX, с"] = round(time.time() - t0, 1)
        times, ok = [], 0
        for _ in range(3):
            t0 = time.time()
            w = eng.synth_chunk(TEXT, TD.CLONE)
            times.append(time.time() - t0)
            ok += int(len(w) > 0 and np.isfinite(w).all())
        dur = len(w) / SR
        res["1. три озвучки подряд"] = {"годных": ok, "время, с": [round(x, 2) for x in times], "звука, с": round(dur, 2)}
        try:
            _in_mlx_thread(lambda: (_ for _ in ()).throw(RuntimeError("искусственный сбой")))
        except RuntimeError:
            pass
        w2 = eng.synth_chunk(TEXT, TD.CLONE)
        res["2. озвучка после сбоя"] = bool(len(w2) > 0)
        wav = Path(tempfile.mkdtemp()) / "check.wav"
        sf.write(str(wav), w2, SR)
        errs, heard = [], {}

        def listen():
            try:
                heard.update(asr.transcribe(str(wav), "turbo", language="ru"))
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))
        th = threading.Thread(target=listen)
        th.start()
        w3 = eng.synth_chunk(TEXT, TD.CLONE)
        th.join(300)
        res["3. проверка слов во время озвучки"] = {"озвучено": bool(len(w3) > 0), "распознано": bool(heard.get("text")),
                                                     "ошибки": errs, "no Stream": any("Stream" in e for e in errs)}
        res["нить MLX одна"] = mlxthread.thread_ident()
        rtf_mlx = min(times) / max(dur, 0.1)
        res["RTF MLX"] = round(rtf_mlx, 3)
        if not a.no_cpu:
            os.environ["HERALD_DEVICE"] = "cpu"
            cpu = engines.get_engine("f5torch")
            t0 = time.time()
            cpu.load()
            res["загрузка F5 torch на процессоре, с"] = round(time.time() - t0, 1)
            t0 = time.time()
            wc = cpu.synth_chunk(TEXT, TD.CLONE)
            tc = time.time() - t0
            rtf_cpu = tc / max(len(wc) / SR, 0.1)
            res["RTF процессор"] = round(rtf_cpu, 3)
            res["процессор медленнее MLX, раз"] = round(rtf_cpu / max(rtf_mlx, 1e-6), 1)
    finally:
        lock.release()
        res["конец"] = time.strftime("%H:%M:%S")
    out = HERE.parent / "bench_out"          # не в git
    out.mkdir(exist_ok=True)
    (out / "mlx_live_14.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    bad = res["1. три озвучки подряд"]["годных"] != 3 or not res["2. озвучка после сбоя"] \
        or res["3. проверка слов во время озвучки"]["ошибки"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
