#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Слепой набор из нескольких голосов (3.6): все под случайными буквами.

Каждый файл - одинаковая обработка (HD_FILTER) и громкость (-18 LUFS), mp3
192 кбит/с (build_blind_set.finish). Ключ «буква -> голос» - в docs/, в вывод не
печатается.

    python "Инструменты разработки/build_set.py" --folder "Сравнение 3.6 - …" \\
        --key "docs/ключ 3.6 - … (не открывать до оценки).md" --title "…" --what "…" \\
        --item "Piper denis" a.wav --item "Vosk 3" b.wav …
"""
from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

LETTERS = list("БВГДЕЖИКЛМНОПРСТУФ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--what", required=True)
    ap.add_argument("--item", nargs=2, action="append", required=True, metavar=("ЧТО", "WAV"))
    a = ap.parse_args()
    from build_blind_set import finish
    letters = sorted(random.SystemRandom().sample(LETTERS, len(a.item)))
    random.SystemRandom().shuffle(a.item)
    os.makedirs(a.folder, exist_ok=True)
    pairs = list(zip(letters, a.item))
    for L, (_name, wav) in pairs:
        finish(wav, os.path.join(a.folder, f"Вариант {L}.mp3"))
    key = [f"# Ключ: {a.title} - НЕ ОТКРЫВАТЬ ДО ОЦЕНКИ\n", "| буква | что |", "|---|---|"]
    key += [f"| {L} | {name} |" for L, (name, _w) in pairs]
    key += ["", a.what]
    open(a.key, "w", encoding="utf-8").write("\n".join(key) + "\n")
    rows = "".join(f"| Вариант {L} | | | |\n" for L in letters)
    open(os.path.join(a.folder, "Оценка.md"), "w", encoding="utf-8").write(
        f"# {a.title}\n\n{a.what}\n\nКлюч - в docs, открывать после оценки.\n\n"
        "| файл | естественность (1-5) | разборчивость (1-5) | включить в Herald? (да / нет) |\n"
        "|---|---|---|---|\n" + rows + "\nЛучшие: ___\n")
    print("собрано:", a.folder, f"({len(letters)} вариантов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
