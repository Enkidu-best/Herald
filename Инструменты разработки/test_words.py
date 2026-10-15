#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сверка слов куска с распознанным (3.2) - без модели, за секунду.

Случаи взяты из замера: что распознаватель путает сам (безударные окончания,
выдумка «продолжение следует» на тишине) - не дефект; пропавшее слово, невнятное
слово и съеденный конец последнего слова - дефект.

    python "Инструменты разработки/test_words.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
import sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

fails: list[str] = []


def must(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        fails.append(msg)


def main() -> int:
    from core.pipeline import _same_word, _ends_like, _bad_words, _edges_verdict
    print("Слово в середине куска:")
    for w, h in [("носителя", "носители"), ("серией", "серии"), ("нажатий", "нажатия"),
                 ("поле", "поля")]:
        must(_same_word(w, h), f"«{w}» ~ «{h}»: путаница распознавателя, не дефект")
    for w, h in [("вернется", "вернеца"), ("огляделся", "крылся"), ("поле", "ли")]:
        must(not _same_word(w, h), f"«{w}» != «{h}»: искажено - дефект")
    h = lambda s: [(x, 0.0, 0.0) for x in s.split()]                          # noqa: E731
    t = "Он вошёл в комнату и огл+ядел+ся. Было темно. Эксперв+ит молч+ал."
    must(_bad_words(h("он вошел в комнату было темно экспервит молчал"), t) == ["огляделся"],
         "пропавшее слово - дефект")
    must(_bad_words(h("он вошел в комнату и огляделся было темно экстер вид молчал"), t) == [],
         "выдуманное имя книги (нет в словаре) не сверяется")
    print("Последнее слово:")
    must(_ends_like("быстрее", "улетаем быстрее продолжение следует".split()),
         "выдумка Whisper после последнего слова не мешает")
    must(not _ends_like("ждать", "просто устал жда".split()), "съеденный конец - дефект")
    must(not _ends_like("интриг", "нескончаемых интри".split()), "«интри» вместо «интриг» - дефект")
    must(_edges_verdict(h("в голову не выйди"), "- В г+олову: - Не в+ыйде...", 5.0),
         "оборванное автором слово («выйде…») сверяется мягко")
    print("\nOK: сверка слов" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    # без разбора объектов onnxruntime/MLX: на Mac он падает с «recursive_mutex»
    _os._exit(rc)
