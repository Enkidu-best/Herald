#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 12: корзина (core/trash) и поправки ударений книги (core/bookfix) - без моделей.

  * в корзину и обратно: файлы на месте, «Заменить?» при занятом месте, «Удалить навсегда»,
    старше 30 дней - удаляется само;
  * поправка ударения: только это слово, все вхождения, регистр и «ё» сохраняются; нет
    поправок - текст тот же (отпечаток «Продолжить» книги не меняется).
Корзина и поправки - во временной папке (настоящие не трогаются).

    python "Инструменты разработки/test_trash_fix.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import tempfile
import time

fails = []


def check(name, ok, got=""):
    print(f"  {'OK  ' if ok else 'ПЛОХО'}  {name}" + (f"   ({got})" if not ok else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    from core import trash, bookfix
    tmp = tempfile.mkdtemp(prefix="herald_t12_")
    trash.TRASH_DIR = os.path.join(tmp, "trash")
    bookfix.FIXES = os.path.join(tmp, "fixes.json")

    print("корзина:")
    vd = os.path.join(tmp, "voices")
    os.makedirs(vd)
    files = [os.path.join(vd, "Тест" + e) for e in (".wav", ".txt", ".fx.json")]
    for i, f in enumerate(files):
        open(f, "w").write("x" * (i + 1) * 100)
    it = trash.put(files, "voice", "Тест")
    check("файлы ушли в корзину", it is not None and not any(os.path.exists(f) for f in files))
    check("запись видна, размер посчитан", [x.id for x in trash.items()] == [it.id] and it.size == 600,
          (trash.items(), it.size))
    trash.restore(it)
    check("«Вернуть» - файлы на месте, запись исчезла",
          all(os.path.exists(f) for f in files) and not trash.items())
    it = trash.put(files, "voice", "Тест")
    open(files[0], "w").write("новый")
    check("место занято - видно до возврата", trash.conflicts(it) == [files[0]])
    try:
        trash.restore(it)
        check("без «Заменить» занятое место не перезаписывается", False)
    except FileExistsError:
        check("без «Заменить» занятое место не перезаписывается",
              open(files[0]).read() == "новый")
    trash.restore(it, replace=True)
    check("«Заменить» - вернулось прежнее", open(files[0]).read() == "x" * 100)
    it = trash.put(files, "voice", "Тест")
    trash.purge(it)
    check("«Удалить навсегда»", not trash.items() and not any(os.path.exists(f) for f in files))
    for f in files:
        open(f, "w").write("x")
    it = trash.put(files, "voice", "Старое")
    import json
    meta = os.path.join(trash.TRASH_DIR, it.id, trash.META)
    d = json.load(open(meta, encoding="utf-8"))
    d["deleted"] = time.time() - 31 * 86400
    json.dump(d, open(meta, "w", encoding="utf-8"))
    check("старше 30 дней - удаляется само", trash.purge_old() == 1 and not trash.items())
    svc = os.path.join(tmp, "Книга", "_служебное")
    os.makedirs(svc)
    open(os.path.join(svc, "текст.txt"), "w").write("t")
    it = trash.put([svc], "service", "Книга")
    import shutil
    shutil.rmtree(os.path.join(tmp, "Книга"))
    try:
        trash.restore(it)
        check("папки книги нет - вернуть некуда", False)
    except FileNotFoundError:
        check("папки книги нет - вернуть некуда", True)

    print("поправка ударения книги:")
    book = os.path.join(tmp, "книга.txt")
    text = "Старый З+амок над рекой. В з+амке тихо; з+амок закрыт. Ёлка и ёлки."
    check("нет поправок - текст тот же", bookfix.apply(text, bookfix.load(book)) == text)
    bookfix.set_fix(book, "замок", 3)
    got = bookfix.apply(text, bookfix.load(book))
    check("все вхождения слова, регистр сохранён", got.count("Зам+ок") == 1 and got.count("зам+ок") == 1, got)
    check("другие слова не тронуты", "з+амке" in got and "Ёлка" in got, got)
    bookfix.set_fix(book, "елка", 0)
    got = bookfix.apply(text, bookfix.load(book))
    check("«ё» в слове сохраняется", "+Ёлка" in got and "ёлки" in got, got)
    check("другая книга - без поправок", bookfix.load(os.path.join(tmp, "другая.txt")) == {})
    bookfix.set_fix(book, "замок", None)
    bookfix.set_fix(book, "ёлка", None)
    check("«Как предложила программа» - поправки сняты", bookfix.load(book) == {})
    check("варианты - по гласным", bookfix.variants("за́мок") == [1, 3])
    shutil.rmtree(tmp, ignore_errors=True)
    print("всё в порядке" if not fails else f"ПРОВАЛЕНО: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
