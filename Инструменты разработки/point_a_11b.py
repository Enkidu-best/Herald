#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 1 (точка А) и базовая линия: один абзац «Главы первой» (абзацы 2-4,
~55 с, с репликами) полным конвейером книги (convert_book: разметка ударений, паузы из
Настроек, проверка слов и пересчёт) - для каждого чтеца:
  F5        - «Основной» профиль, как сейчас (базовая линия);
  Qwen      - обученный голос как сейчас (общий atempo fx.json speed 0,86 / 0,84);
  а         - обученный без подгонки темпа;
  б         - темп каждого куска к темпу живого чтеца (gen.json tempo_fit, без общего atempo);
  в         - кусок 1-2 предложения (до 220 знаков), без подгонки темпа.
Пишет «Сравнение 3.10 - лучшие варианты/Точка А/<чтец> - <вариант>.mp3» и «Как сделано.md».
Временные голоса «<чтец> (обучен а/б)» убираются в конце. Под ../.heavy-lock.

    python "Инструменты разработки/point_a_11b.py" [F5 Qwen а б в]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import json
import shutil
import time

SRC_ALL = "Архив/голоса диктора 3.9/длинный кусок/Глава первая (отрывок).txt"
TEXT = "Архив/11б/абзац.txt"
DST = "Сравнение 3.10 - лучшие варианты/Точка А"
READERS = {"Игорь Ященко": ("Ященко", 5.82), "Вячеслав Герасимов": ("Герасимов", 5.64)}


def paragraph() -> str:
    _os.makedirs(_os.path.dirname(TEXT), exist_ok=True)
    if not _os.path.exists(TEXT):
        p = [x for x in open(SRC_ALL, encoding="utf-8").read().split("\n\n") if x.strip()]
        open(TEXT, "w", encoding="utf-8").write("\n\n".join(p[2:5]) + "\n")   # 0, 1 - «Глава первая»
    return TEXT


def temp_voice(reader: str, tag: str, gen: dict) -> str:
    from core.engines.mlxaudio import own_voices_dir
    d = own_voices_dir()
    base = _os.path.join(d, f"{reader} (обучен)")
    name = f"{reader} (обучен {tag})"
    for ext in (".wav", ".txt", ".model"):
        shutil.copy(base + ext, _os.path.join(d, name + ext))
    json.dump(gen, open(_os.path.join(d, name + ".gen.json"), "w", encoding="utf-8"))
    return name


def drop_voice(name: str) -> None:
    from core.engines.mlxaudio import own_voices_dir
    for f in glob.glob(_os.path.join(own_voices_dir(), glob.escape(name) + ".*")):
        _os.remove(f)


def main() -> int:
    import core.pipeline as P
    from core.profiles import PROFILES, engine_kwargs
    from core.engines.mlxaudio import trained_gen
    src = paragraph()
    want = _sys.argv[1:] or ["F5", "Qwen", "а", "б", "в"]
    _os.makedirs(DST, exist_ok=True)
    made, notes = [], []
    for reader, (short, target) in READERS.items():
        base_gen = trained_gen(f"{reader} (обучен)")
        plan = {
            "F5": ("main", reader, {}, "F5 «Основной», как сейчас"),
            "Qwen": ("qwen", f"{reader} (обучен)", {}, "обученный Qwen как сейчас (общий atempo)"),
            "а": ("qwen", ("а", base_gen), {}, "обученный, без подгонки темпа"),
            "б": ("qwen", ("б", {**base_gen, "tempo_fit": target}), {},
                  f"обученный, темп каждого куска к живому чтецу ({target} слога/с), без общего atempo"),
            "в": ("qwen", ("а", base_gen), {"max_chars": 220}, "обученный, кусок 1-2 предложения, без подгонки"),
        }
        for v in want:
            prof, voice, over, about = plan[v]
            tmp = None
            if isinstance(voice, tuple):
                tmp = voice = temp_voice(reader, *voice)
            p = PROFILES[prof]
            kw = {**engine_kwargs(p), **over}
            work = _os.path.abspath(f"Архив/11б/точка А/{short} - {v}")
            shutil.rmtree(work, ignore_errors=True)
            opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60, make_m4b=False,
                                    make_chapter_mp3=True, sound_check=True, narrator=reader)
            t0 = time.time()
            res = P.convert_book(src, engine=p.engine, eng_kwargs=kw, voice=voice, out_dir=work, options=opts)
            mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
            out = _os.path.join(DST, f"{short} - {v}.mp3")
            shutil.copy(mp3, out)
            made.append(out)
            notes.append(f"| {short} - {v}.mp3 | {about} | {p.engine}, {json.dumps(kw, ensure_ascii=False)} | "
                         f"{res.duration:.1f} с | {time.time() - t0:.0f} с |")
            print(out, f"{res.duration:.1f} с", flush=True)
            P.unload_engines()
            if tmp:
                drop_voice(tmp)
    with open(_os.path.join(DST, "Как сделано.md"), "w", encoding="utf-8") as f:
        f.write("# Точка А (11б, шаг 1)\n\nСделано полным конвейером: да (convert_book - разметка ударений, "
                "паузы из Настроек, проверка слов и пересчёт). Текст - «Глава первая», абзацы 2-4 "
                "(`Архив/11б/абзац.txt`).\n\n| файл | что | движок и настройки | звук | счёт |\n|---|---|---|---|---|\n")
        f.write("\n".join(notes) + "\n")
    print(json.dumps(made, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
