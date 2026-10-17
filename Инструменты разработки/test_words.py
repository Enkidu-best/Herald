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
    for word, heard, post in [("и", "", -9.0), ("в", "", -12.0)]:
        must(not is_bad(w(word, heard, post)),
             f"однобуквенное «{word}» не судим: в беглой речи сливается с соседом (3.7)")
    must(is_bad(w("он", "о", -4.69)), "двухбуквенное «он» - по-прежнему судим")
    for word, heard, post in [("абориген", "обреген", -1.79), ("судна", "судно", -1.04),
                              ("сетчатку", "сичатку", -1.07), ("щёлкнул", "щелкнул", -0.3)]:
        must(not is_bad(w(word, heard, post)),
             f"«{word}» -> «{heard}» ({post}): безударные гласные - не ошибка")
    print("Края куска по звуку (3.5, вместо Whisper):")
    from core.align import edges_ok

    def ch(*pairs):
        return [w(word, heard, post) for word, heard, post in pairs]
    must(edges_ok(ch(("Прости", "прости", -0.1), ("ждать", "ждать", -0.2)),
                  "- Прости. - Я устал ждать."), "оба края целы")
    must(not edges_ok(ch(("Прости", "прости", -0.1), ("ждать", "жда", -0.6)),
                      "- Прости. - Я устал ждать."), "«ждать» -> «жда»: съеден конец")
    must(not edges_ok(ch(("Это", "это", -0.1), ("интриг", "интри", -0.5)),
                      "Это поле для интриг."), "«интриг» -> «интри»: съеден конец")
    must(edges_ok(ch(("Это", "это", -0.1), ("интриг", "интрик", -0.4)),
                  "Это поле для интриг."), "оглушение на конце («интрик») - не ошибка")
    must(not edges_ok(ch(("Такая", "такая", -0.1), ("выйдет", "вредит", -0.6)),
                      "Такая мысль в голову не выйдет."), "«выйдет» -> «вредит»: слово заменено")
    must(edges_ok(ch(("Голову", "голову", -0.1), ("выйде", "выйди", -0.4)),
                  "- В голову: - Не выйде..."), "многоточие автора: конец не сверяется")
    must(edges_ok(ch(("Ветер", "ветер", -0.1), ("берегу", "берегу", -0.3)),
                  "Ветер дул к берегу."), "безударная гласная на конце - не ошибка")
    must(not edges_ok(ch(("Перед", "", -9.0), ("берегу", "берегу", -0.3)),
                      "Перед берегу."), "первое слово съедено")
    print("Ударение по звуку - какие слова проверять (3.8):")
    import tempfile
    from core import stresscheck as SC
    must(SC.risky("+отмель") and SC.risky("п+очерком"), "редкие слова Андрея - под проверкой")
    must(not SC.risky("мол+око") and not SC.risky("к+омната"), "частые слова - без проверки")
    must(SC.risky("дор+ога") == 2, "омограф (дорóга / дóрога) - под проверкой")
    must(not SC.risky("+он") and not SC.risky("ещё"), "односложные и с «ё» - без проверки")
    must(SC.risky("мол+око", {"молоко": 1}) == 3, "слово, трудное для модели, - проверяется первым")
    must(SC._parts("С+еверная +отмель, сл+ово-в-сл+ово.") ==
         ["С+еверная", "+отмель", "сл+ово", "в", "сл+ово"], "слова - как у выравнивателя")
    keep = SC.hard_path
    tmp = tempfile.mkdtemp()
    SC.hard_path = lambda: _os.path.join(tmp, "трудные.json")
    try:
        SC.remember_hard("модель-1", ["отмель", "отмель", "каторжник"])
        must(SC.hard_words("модель-1") == {"отмель": 2, "каторжник": 1}
             and SC.hard_words("модель-2") == {}, "трудные слова - по ключу модели, с повторами")
        must(SC.calibrated("модель-1") and not SC.calibrated("модель-2"),
             "новая модель - тестовый набор ещё не прогнан")
    finally:
        SC.hard_path = keep
    must(len(SC.test_set()) >= 20 and "+отмель" in SC.test_set(), "тестовый набор на месте")
    print("Растянутая гласная (3.4, «никогооо»):")
    import numpy as np
    from core.align import stretched
    sr = 16000

    def take(stretch):
        # «нет никого, кто»: голос 0,15 с на слово, после «никого» - stretch с звука
        tone = lambda sec: (0.3 * np.sin(np.arange(int(sec * sr)) / 3)).astype(np.float32)  # noqa: E731
        pause = lambda sec: np.zeros(int(sec * sr), np.float32)  # noqa: E731
        a = np.concatenate([tone(0.3), pause(0.1), tone(0.3 + stretch), pause(0.15), tone(0.3)])
        ws = [WordScore("нет", "нет", 0.0, 0.25, 0, 0, 0, 0, ""),
              WordScore("никого", "никого", 0.4, 0.65, 0, 0, 0, 0, ","),
              WordScore("кто", "кто", 0.4 + 0.3 + stretch + 0.15, 1.3 + stretch, 0, 0, 0, 0, "")]
        return stretched(ws, a, sr)
    must(take(0.15) == [], "обычная конечная гласная перед запятой - не ошибка")
    must(take(0.9) == ["никого"], "гласная тянется почти секунду - пересчитать кусок")
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
