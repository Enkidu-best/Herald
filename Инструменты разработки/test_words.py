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
    print("Проверка по звуку (3.3, случаи из размеченного набора):")
    # «думаешь» -> «думать» ловится через соседа: «-ться» модель склеила со
    # следующим «что» (на его месте слышно «ся», -13) - кусок всё равно помечен
    from core.align import WordScore, is_bad
    def w(word, heard, post):
        return WordScore(word, word.lower(), 0, 0, 0, 0, 0, 0, "", [], post, 0, heard, 0)
    for word, heard, post in [("ногой", "новое", -1.02), ("что", "ся", -13.1),
                              ("на", "", -7.14), ("нажатий", "наж", -4.96), ("он", "о", -4.69)]:
        must(is_bad(w(word, heard, post)), f"«{word}» -> «{heard}» ({post}): ошибка Андрея поймана")
    for word, heard, post in [("абориген", "обреген", -1.79), ("судна", "судно", -1.04),
                              ("сетчатку", "сичатку", -1.07), ("щёлкнул", "щелкнул", -0.3)]:
        must(not is_bad(w(word, heard, post)),
             f"«{word}» -> «{heard}» ({post}): безударные гласные - не ошибка")
    from core.pipeline import _clone
    must(_clone(type("E", (), {"name": "f5mlx"})()) and _clone(type("E", (), {"name": "f5torch"})()),
         "клон F5 проверяется по звуку")
    must(not _clone(type("E", (), {"name": "silero"})()),
         "готовые голоса Silero по звуку не проверяются (дольше их счёта)")
    print("\nOK: сверка слов" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    # без разбора объектов onnxruntime/MLX: на Mac он падает с «recursive_mutex»
    _os._exit(rc)
