#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 14 В: всё MLX в процессе - через одну нить (core/mlxthread.py). Без видеокарты и моделей: MLX и
mlx_whisper подменены. Живая проверка (три озвучки подряд, после сбоя, «Прочитать вслух» во время книги) - под
замком, «Инструменты разработки/mlx_live_14.py».

    python "Инструменты разработки/test_mlxthread.py"
"""
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import asr, mlxthread  # noqa: E402

fails = []


def check(name, ok, note=""):
    print(("  OK    " if ok else "  ПАДАЕТ ") + name + (f": {note}" if note and not ok else ""), flush=True)
    if not ok:
        fails.append(name)


# 1. одна нить: вызовы из разных нитей идут в одной и той же нити MLX
seen = []
ths = [threading.Thread(target=lambda: seen.append(mlxthread.run(threading.get_ident))) for _ in range(5)]
for t in ths:
    t.start()
for t in ths:
    t.join()
check("вызовы из 5 нитей - в одной нити MLX", len(set(seen)) == 1 and seen[0] == mlxthread.thread_ident(), str(seen))

# 2. вызов изнутри нити MLX - сразу, без взаимной блокировки
got = mlxthread.run(lambda: mlxthread.run(lambda: "вложенный"))
check("вложенный вызов не ждёт сам себя", got == "вложенный")

# 3. ошибка - кэш MLX очищается в той же нити, ошибка уходит вызывающему
cleared = []
fake_mx = types.SimpleNamespace(clear_cache=lambda: cleared.append(threading.get_ident()))
sys.modules["mlx"] = types.SimpleNamespace(core=fake_mx)
sys.modules["mlx.core"] = fake_mx
try:
    mlxthread.run(lambda: 1 / 0)
    raised = False
except ZeroDivisionError:
    raised = True
check("ошибка - кэш очищен в нити MLX, ошибка дошла", raised and cleared == [mlxthread.thread_ident()], str(cleared))

# 4. проверка слов по звуку (mlx_whisper) - тоже в нити MLX
calls = []
sys.modules["mlx_whisper"] = types.SimpleNamespace(
    transcribe=lambda *a, **k: (calls.append(threading.get_ident()), {"text": "ок"})[1])
asr._backend = "mlx"
out = asr.transcribe("x.wav", "turbo", language="ru")
check("mlx_whisper - в нити MLX", out == {"text": "ок"} and calls == [mlxthread.thread_ident()], str(calls))

# 5. выгрузка моделей (нить сервера /speak) чистит кэш в нити MLX
from core import pipeline  # noqa: E402
cleared.clear()
threading.Thread(target=pipeline.unload_engines).start()
import time  # noqa: E402
time.sleep(0.5)
check("выгрузка - очистка кэша в нити MLX", cleared == [mlxthread.thread_ident()], str(cleared))

print(f"\nИтог: {5 - len(fails)} из 5")
sys.exit(1 if fails else 0)
