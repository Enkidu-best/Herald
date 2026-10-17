#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сценарий 3.7: менеджер голосов по-настоящему - из официальных источников.

  1. Нет интернета - понятная ошибка, а не «URLError».
  2. Piper Joe: образец без загрузки, движок Piper с pypi.org (GPL - не в
     программе), голос с проверкой суммы, фраза, удаление.
  3. Пакет Vosk (747 МБ): загрузка, фраза каждым из трёх голосов, удаление.

Всё - в отдельной временной «домашней папке», голоса и пакеты пользователя не
трогаются. На Mac движок Piper из .venv скрывается, чтобы проверить загрузку
как в установленной программе.

    python "Инструменты разработки/scenario_voices.py"            # всё
    python "Инструменты разработки/scenario_voices.py" --no-vosk  # без пакета Vosk
"""
from __future__ import annotations

import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOME = tempfile.mkdtemp(prefix="herald-voices-")
os.environ["HOME"] = HOME                     # Path.home() - на Mac / Linux
os.environ["USERPROFILE"] = HOME              # и на Windows

import numpy as np

from core import voice_store as VS
from core.engines import fast

FAILED: list[str] = []


def step(name, fn):
    t = time.time()
    try:
        out = fn()
        print(f"  OK    {name} ({time.time() - t:.1f} с){' - ' + out if out else ''}", flush=True)
    except Exception as e:
        FAILED.append(name)
        print(f"  ПАДАЕТ {name}: {type(e).__name__}: {e}", flush=True)


def must(cond, msg):
    if not cond:
        raise AssertionError(msg)


def say(engine, vid: str, text: str) -> str:
    a = engine.synth_chunk(text, vid)
    sec = len(a) / fast.SR
    must(sec > 0.8 and float(np.abs(a).max()) > 0.05, f"{vid}: тишина или пусто ({sec:.2f} с)")
    return f"{vid}: {sec:.1f} с звука"


def no_internet():
    import urllib.request
    orig = urllib.request.urlopen

    def down(*a, **k):
        raise OSError("Network is unreachable")
    urllib.request.urlopen = down
    try:
        VS._PIPER_INDEX = None
        try:
            VS.download(VS.by_id("piper:en_US-joe-medium"))
        except VS.NoInternet as e:
            must("интернет" in str(e), f"непонятное сообщение: {e}")
            return str(e)
        raise AssertionError("без сети загрузка не упала")
    finally:
        urllib.request.urlopen = orig


def main() -> int:
    vosk = "--no-vosk" not in sys.argv
    print(f"домашняя папка сценария: {HOME}")
    orig_has = VS._has_module
    VS._has_module = lambda n: False if n == "piper" else orig_has(n)   # как в установщике

    print("нет интернета:")
    step("понятная ошибка", no_internet)

    print("Piper (Joe):")
    joe = VS.by_id("piper:en_US-joe-medium")
    step("образец без загрузки голоса",
         lambda: f"{os.path.getsize(VS.sample(joe)) // 1024} КБ, голос не скачан: "
                 f"{not VS.installed(joe)}")
    progress = []
    step("скачать (движок с pypi.org + голос, суммы сошлись)",
         lambda: (VS.download(joe, progress=lambda f, t="": progress.append(f)),
                  must(VS.installed(joe), "после загрузки не установлен"),
                  must(progress and max(progress) >= 0.99, "не было хода загрузки"),
                  f"{VS.size_on_disk(joe) / 1e6:.0f} МБ")[-1])
    eng = fast.PiperEngine()
    step("фраза голосом Joe", lambda: (
        say(eng, joe.id, "The lantern on the mast barely glowed."),
        must("piper_engine" in (sys.modules["piper"].__file__ or ""),
             "движок Piper взят не из папки данных"))[0])
    step("удалить", lambda: (VS.delete(joe), must(not VS.installed(joe), "остался"),
                             must(joe.id not in [v.id for v in fast.enabled_voices()],
                                  "остался в списках"))[-1])

    if vosk:
        print("Vosk (пакет на три голоса):")
        v4 = VS.by_id("vosk:4")
        step("скачать пакет (md5 сошлась)",
             lambda: (VS.download(v4), must(VS.installed(v4), "не установлен"),
                      f"{VS.size_on_disk(v4) / 1e6:.0f} МБ на диске")[-1])
        veng = fast.VoskEngine()
        for vid in ("vosk:4", "vosk:3", "vosk:2"):
            step(f"фраза голосом {VS.by_id(vid).title}",
                 lambda vid=vid: say(veng, vid, "Вода у берега была тёмной."))
        step("удалить пакет", lambda: (VS.delete(v4),
                                       must(not any(VS.installed(VS.by_id(x))
                                                    for x in ("vosk:4", "vosk:3", "vosk:2")),
                                            "остался"))[-1])

    import shutil
    shutil.rmtree(HOME, ignore_errors=True)
    print("всё в порядке" if not FAILED else f"\nПРОВАЛЕНО: {len(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
