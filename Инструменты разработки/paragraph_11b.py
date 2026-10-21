#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б: абзац «Главы первой» (Архив/11б/абзац.txt) полным конвейером книги одним
голосом - для контрольных точек (В, В2 и далее).

    python "Инструменты разработки/paragraph_11b.py" <папка> <профиль> <голос> <имя файла> [k=v ...]
    например: ... "Точка В" qwen "Игорь Ященко (обучен Base)" "Ященко - Qwen Base"
Пишет «Сравнение 3.10 - лучшие варианты/<папка>/<имя файла>.mp3» и строку в «Как сделано.md».
Под ../.heavy-lock.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import json
import shutil
import time

TEXT = "Архив/11б/абзац.txt"


def main() -> int:
    import core.pipeline as P
    from core.profiles import PROFILES, engine_kwargs
    folder, prof, voice, name = _sys.argv[1:5]
    over = {}
    for kv in _sys.argv[5:]:
        k, v = kv.split("=", 1)
        try:
            over[k] = json.loads(v)
        except ValueError:
            over[k] = v
    dst = f"Сравнение 3.10 - лучшие варианты/{folder}"
    _os.makedirs(dst, exist_ok=True)
    p = PROFILES[prof]
    engine = over.pop("engine", p.engine)          # engine=f5torch - тот же F5 на torch (пробы 3б)
    kw = {**engine_kwargs(p), **over}
    work = _os.path.abspath(f"Архив/11б/{folder}/{name}")
    shutil.rmtree(work, ignore_errors=True)
    opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60, make_m4b=False,
                            make_chapter_mp3=True, sound_check=True, narrator=voice)
    t0 = time.time()
    res = P.convert_book(TEXT, engine=engine, eng_kwargs=kw, voice=voice, out_dir=work, options=opts)
    mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
    out = _os.path.join(dst, name + ".mp3")
    shutil.copy(mp3, out)
    note = _os.path.join(dst, "Как сделано.md")
    if not _os.path.exists(note):
        open(note, "w", encoding="utf-8").write(
            f"# {folder} (11б)\n\nСделано полным конвейером: да (convert_book - разметка ударений, паузы из "
            "Настроек, проверка слов и пересчёт). Текст - «Глава первая», абзацы 2-4 (`Архив/11б/абзац.txt`).\n\n"
            "| файл | голос | движок и настройки | звук | счёт |\n|---|---|---|---|---|\n")
    with open(note, "a", encoding="utf-8") as f:
        own = " + дообученные веса (HERALD_F5_WEIGHTS)" if _os.environ.get("HERALD_F5_WEIGHTS") else ""
        f.write(f"| {name}.mp3 | {voice} | {engine}{own}, {json.dumps(kw, ensure_ascii=False)} | "
                f"{res.duration:.1f} с | {time.time() - t0:.0f} с |\n")
    print(out, f"{res.duration:.1f} с", flush=True)
    P.unload_engines()
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
