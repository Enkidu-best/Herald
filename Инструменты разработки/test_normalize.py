#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Числа и сокращения: работает именно ru-normalizr, а не запасной разбор (3.2).

На Linux arm64 он молча отваливался (сломанный быстрый словарь pymorphy3), и
«МГУ», инициалы, падежи читались хуже - поймано в Docker. Пять фраз из стенда
bench_normalize (полный стенд - 50 фраз).

    python "Инструменты разработки/test_normalize.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import re

CASES = [
    ("В 1990 году он окончил школу.", "в тысяча девятьсот девяностом году он окончил школу"),
    ("Профессор МГУ выступил.", "профессор эм гэ у выступил"),
    ("Доктор А. П. Чехов уехал.", "доктор а пэ чехов уехал"),
    ("Цена выросла на 15 %.", "цена выросла на пятнадцать процентов"),
    ("Пётр I основал город.", "пётр первый основал город"),
]


def norm(s: str) -> str:
    return re.sub(r"[^\w ]", " ", s.lower().replace("-", " ")).split().__str__()


def main() -> int:
    import core.normalize as N
    fails = []
    for src, want in CASES:
        got = N.normalize_text(src)
        ok = norm(got) == norm(want)
        print(("  ✓ " if ok else "  ✗ ") + f"{src} -> {got}")
        if not ok:
            fails.append(src)
    if N._RNZ_FAILED:
        print("  ✗ ru-normalizr не работает - числа разбирает запасной способ")
        fails.append("ru-normalizr")
    print("\nOK: нормализация" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
