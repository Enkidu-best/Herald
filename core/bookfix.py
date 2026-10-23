# -*- coding: utf-8 -*-
"""Поправка ударения в спорном слове ОДНОЙ книги (задание 12, п. 5).

В «Как будет прочитано» подсвечены слова, где два расстановщика ударений разошлись.
Щелчок по такому слову - выбрать гласную; выбор действует только на эту книгу и
ставится после всех шагов ударений (RUAccent, silero-stress, справочник, единое
ударение по книге) - на каждое вхождение слова. Это не ручной словарь (его Андрей
отверг): своё слово завести нельзя, правятся только спорные слова этой книги.

Хранится вне папки книги (служебные файлы книги уходят в корзину, поправки - нет):
`<папка данных>/book_fixes.json` = {путь книги: {слово: номер буквы ударной гласной}}.
"""
from __future__ import annotations

import json
import os
import re

from .osutil import data_dir as _data_dir
FIXES = os.path.join(str(_data_dir()), "book_fixes.json")
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_WORD = re.compile(r"[А-Яа-яЁё+́]+")


def key(word: str) -> str:
    """Слово без ударения, нижний регистр, ё = е."""
    return word.replace("+", "").replace("́", "").lower().replace("ё", "е")


def _all() -> dict:
    try:
        with open(FIXES, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load(book: str) -> dict[str, int]:
    return dict(_all().get(os.path.abspath(book), {}))


def set_fix(book: str, word: str, letter: int | None) -> None:
    """Ударение на букву `letter` (номер в слове без знаков); None - как предложила программа."""
    data = _all()
    b = os.path.abspath(book)
    fixes = data.get(b, {})
    if letter is None:
        fixes.pop(key(word), None)
    else:
        fixes[key(word)] = int(letter)
    if fixes:
        data[b] = fixes
    else:
        data.pop(b, None)
    os.makedirs(os.path.dirname(FIXES), exist_ok=True)
    tmp = FIXES + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, FIXES)


def variants(word: str) -> list[int]:
    """Номера букв-гласных слова (без знаков) - варианты ударения."""
    plain = word.replace("+", "").replace("́", "")
    return [i for i, c in enumerate(plain) if c in VOWELS]


def marked(word: str, letter: int) -> str:
    """Слово со знаком «+» перед гласной номер `letter`."""
    plain = word.replace("+", "").replace("́", "")
    return plain[:letter] + "+" + plain[letter:]


def apply(text: str, fixes: dict[str, int]) -> str:
    """Поставить поправки во всём тексте (формат «+» перед ударной гласной)."""
    if not fixes:
        return text

    def sub(m):
        w = m.group(0)
        i = fixes.get(key(w))
        plain = w.replace("+", "").replace("́", "")
        if i is None or i >= len(plain) or plain[i] not in VOWELS:
            return w
        return marked(plain, i)
    return _WORD.sub(sub, text)
