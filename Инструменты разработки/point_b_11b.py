#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 2 (точка Б): F5 «Основной» с ВОССТАНОВЛЕННЫМ образцом чтеца (restore_ref_11b.py:
Sidon, DFN, Sidon70) против F5 как сейчас - тот же абзац, полным конвейером книги.
Пишет «Сравнение 3.10 - лучшие варианты/Точка Б/<чтец> - F5 <образец>.mp3». Под ../.heavy-lock.

    python "Инструменты разработки/point_b_11b.py" [Sidon DFN Sidon70]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import shutil

TEXT = "Архив/11б/абзац.txt"
DST = "Сравнение 3.10 - лучшие варианты/Точка Б"
READERS = {"Игорь Ященко": "Ященко", "Вячеслав Герасимов": "Герасимов"}


def main() -> int:
    import core.pipeline as P
    from core.profiles import PROFILES, engine_kwargs
    _os.makedirs(DST, exist_ok=True)
    p = PROFILES["main"]
    for reader, short in READERS.items():
        for tag in _sys.argv[1:] or ["Sidon", "DFN", "Sidon70"]:
            voice = f"{reader} ({tag})"
            work = _os.path.abspath(f"Архив/11б/точка Б/{short} - {tag}")
            shutil.rmtree(work, ignore_errors=True)
            opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60, make_m4b=False,
                                    make_chapter_mp3=True, sound_check=True, narrator=reader)
            res = P.convert_book(TEXT, engine=p.engine, eng_kwargs=engine_kwargs(p), voice=voice, out_dir=work,
                                 options=opts)
            mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
            shutil.copy(mp3, _os.path.join(DST, f"{short} - F5 {tag}.mp3"))
            print(short, tag, f"{res.duration:.1f} с", flush=True)
            P.unload_engines()
        shutil.copy(f"Сравнение 3.10 - лучшие варианты/Точка А/{short} - F5.mp3", _os.path.join(DST, f"{short} - F5 как сейчас.mp3"))
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
