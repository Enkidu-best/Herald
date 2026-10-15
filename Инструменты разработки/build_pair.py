#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Слепая пара для прослушивания: два варианта одного текста под случайными буквами.

Оба варианта - одинаковая обработка (HD_FILTER) и громкость (-18 LUFS), mp3
192 кбит/с (build_blind_set.finish). Ключ «буква -> вариант» пишется в docs/,
в вывод не печатается.

    python "Инструменты разработки/build_pair.py" --folder "Сравнение 3.3 - глава" \\
        --key "docs/ключ 3.3 - глава (не открывать до оценки).md" --title "…" \\
        --a "3.2 без проверки" а.wav --b "3.3 с проверкой" б.wav --what "…"
"""
from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--what", required=True)
    ap.add_argument("--a", nargs=2, required=True, metavar=("ЧТО", "WAV"))
    ap.add_argument("--b", nargs=2, required=True, metavar=("ЧТО", "WAV"))
    ap.add_argument("--columns", default="естественность (1-5) | разборчивость слов (1-5) | что съедено или смазано")
    a = ap.parse_args()
    from build_blind_set import finish
    letters = random.SystemRandom().sample(list("БВГДЕИЛОПС"), 2)
    os.makedirs(a.folder, exist_ok=True)
    pairs = list(zip(letters, [a.a, a.b]))
    for L, (_name, wav) in pairs:
        finish(wav, os.path.join(a.folder, f"Вариант {L}.mp3"))
    key = [f"# Ключ: {a.title} - НЕ ОТКРЫВАТЬ ДО ОЦЕНКИ\n", "| буква | что |", "|---|---|"]
    key += [f"| {L} | {name} |" for L, (name, _w) in sorted(pairs)]
    key += ["", a.what]
    open(a.key, "w", encoding="utf-8").write("\n".join(key) + "\n")
    cols = [c.strip() for c in a.columns.split("|")]
    head = "| файл | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = "".join(f"| Вариант {L} |" + " |" * len(cols) + "\n" for L in sorted(letters))
    open(os.path.join(a.folder, "Оценка.md"), "w", encoding="utf-8").write(
        f"# {a.title}\n\n{a.what}\n\nКлюч - в docs, открывать после оценки.\n\n{head}{rows}\n"
        "Лучше: ___ (или «не отличить»)\n")
    print("собрано:", a.folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
