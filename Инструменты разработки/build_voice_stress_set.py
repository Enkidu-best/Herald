#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Короткий слепой набор «наш голос + верные ударения» (3.8, после оценки 19.10).

4 варианта × 2 чтеца, 10 фраз (Архив/qwen голос 3.8/<режим>/): F5 как сейчас,
Qwen Base + адаптер ru-stress на половину силы, Qwen ru-stress -> kNN-VC, то же
с адаптером 0,5 -> kNN-VC. Номера записей случайные, громкость выровнена; лист
оценок Excel и инструкция - как в основном наборе 3.8; ключ - отдельным файлом.

    python "Инструменты разработки/build_voice_stress_set.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import importlib.util
import json
import random
import shutil
import subprocess

import numpy as np

SRC = "Архив/qwen голос 3.8"
FOLDER = "Сравнение 3.8 - голос и ударения"
KEY = "docs/ключ 3.8 - голос и ударения (не открывать до конца прослушивания).md"
VARIANTS = {"f5": "F5 Misha - как сейчас",
            "lora05": "Qwen3-TTS Base + адаптер ru-stress на половину силы",
            "rs_xvec_knnvc": "Qwen ru-stress (вектор голоса) -> kNN-VC по записям чтеца",
            "lora05_knnvc": "Qwen Base + адаптер 0,5 -> kNN-VC по записям чтеца"}
VOICES = ("Вячеслав Герасимов", "Игорь Ященко")


def main() -> int:
    import soundfile as sf
    spec = importlib.util.spec_from_file_location("e", "Инструменты разработки/bench_engines_38.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    phr = json.load(open(_os.path.join(SRC, "фразы.json"), encoding="utf-8"))
    recs = [(n, v) for n in VARIANTS for v in VOICES]
    random.Random("Herald 3.8, голос и ударения").shuffle(recs)
    shutil.rmtree(FOLDER, ignore_errors=True)
    _os.makedirs(FOLDER)
    for num, (n, v) in enumerate(recs, 1):
        parts, sr0 = [], 48000
        for i in range(len(phr)):
            f = _os.path.join(SRC, n, f"{v}__{i + 1:02d}.wav")
            if not _os.path.exists(f):
                parts.append(np.zeros(int(1.5 * sr0), np.float32))
                continue
            au, sr = sf.read(f, dtype="float32")
            if sr != sr0:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, sr0)
                au = resample_poly(au, sr0 // g, sr // g).astype(np.float32)
            parts += [e._level(au), np.zeros(int(1.5 * sr0), np.float32)]
        tmp = _os.path.join(SRC, f"набор {num:02d}.wav")
        sf.write(tmp, np.concatenate(parts), sr0)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-b:a", "256k",
                        _os.path.join(FOLDER, f"Запись {num:02d}.mp3")], check=True)
    for v in VOICES:
        shutil.copy(_os.path.join("Сравнение 3.8 - движки", f"Образец чтеца - {v}.mp3"), FOLDER)
    e.FOLDER = FOLDER
    rows = [(p["phrase"], p["word"]) for p in phr]
    e.sheet(recs, rows)
    text = open(_os.path.join("Сравнение 3.8 - движки", "Как слушать.md"), encoding="utf-8").read()
    import re as _re
    text = _re.sub(r"\*\*Что это\.\*\* \d+ записей", f"**Что это.** {len(recs)} записей", text)
    text = (text.replace("одни и те же 20 фраз разными движками синтеза",
                         "одни и те же 10 фраз разными способами синтеза")
                .replace("- фразы 1-17 - короткие", "- все 10 - короткие")
                .replace("**Ключ** (какой движок под каким номером)", "**Ключ** (какой вариант под каким номером)")
                .replace("«ключ 3.8 - движки (не открывать до конца прослушивания).md»",
                         "«ключ 3.8 - голос и ударения (не открывать до конца прослушивания).md»"))
    text = "\n".join(l for l in text.splitlines() if "фразы 18-20" not in l and "Достаточно прослушать" not in l
                     and "явно хуже других" not in l)
    text = text.replace("# Как слушать: слепое сравнение движков Herald 3.8",
                        "# Как слушать: голос и ударения вместе (Herald 3.8, второй набор)")
    open(_os.path.join(FOLDER, "Как слушать.md"), "w", encoding="utf-8").write(text + "\n")
    m = {}
    for part in ("замер.json", "замер lora.json", "замер knnvc.json"):
        try:
            m.update(json.load(open(_os.path.join(SRC, part), encoding="utf-8")))
        except OSError:
            pass
    key = ["# Ключ 3.8 - голос и ударения", "", "Открывать после того, как лист заполнен.", "",
           "| запись | вариант | чтец | похожесть на образец (ECAPA) | постоянство | ударения (meter2, оба чтеца) |",
           "|---|---|---|---|---|---|"]
    for num, (n, v) in enumerate(recs, 1):
        r = m.get(n, {})
        key.append(f"| {num:02d} | {VARIANTS[n]} | {v} | {r.get('sim', {}).get(v, float('nan')):.2f} | "
                   f"{r.get('pair', {}).get(v, float('nan')):.2f} | "
                   f"{r.get('stress', ['-', '-'])[0]} из {r.get('stress', ['-', '-'])[1]} |")
    open(KEY, "w", encoding="utf-8").write("\n".join(key) + "\n")
    print(f"{len(recs)} записей в «{FOLDER}»")
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
