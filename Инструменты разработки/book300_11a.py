#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, п. 6: 300 случайных слов из трёх книг Андрея - набор для замера ударений.

По 100 слов на книгу: случайное предложение (40-260 знаков), в нём случайное слово
из двух и больше слогов без «ё», не имя (строчное или первое в предложении). Рядом -
соседние предложения и весь абзац (для разметки «как раньше» - целым куском).
Верное ударение - отдельным шагом (`--truth`): где словарь по Зализняку
(CC0, DataverseNO doi:10.18710/AAFCJP) знает форму однозначно - по нему; где нет
или слово - омограф - по разбору вручную (файл «разбор.tsv»).
Набор - в Архив/ (в нём фразы из чужих книг, в git не идёт).

    python "Инструменты разработки/book300_11a.py" build
    python "Инструменты разработки/book300_11a.py" truth Архив/…/z.tsv
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import random
import re

OUT = "Архив/ударения 11а"
BOOKS = [
    "Книги/Жажда власти 1 - Сергей Тармашев/Жажда власти. Книга 1.fb2",
    "Образцы голосов/Артем Каменистый.fb2",
    _os.path.expanduser("~/Documents/Torrents/Сергей Тармашев - Сборник произведений/Холод/"
                        "01.Неотвратимая гибель.fb2"),
]
VOW = "аеёиоуыэюя"
WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")


def build() -> int:
    from razdel import sentenize
    from core.readers import read_document
    rnd = random.Random(11)
    items = []
    for bi, path in enumerate(BOOKS):
        doc = read_document(path)
        text = doc.text or "\n".join(c.text for c in (doc.chapters or []))
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 60]
        cands = []
        for pi, p in enumerate(paras):
            ss = [s.text for s in sentenize(p)]
            for si, s in enumerate(ss):
                if 40 <= len(s) <= 260:
                    cands.append((pi, si, ss))
        rnd.shuffle(cands)
        got, used = 0, set()
        for pi, si, ss in cands:
            if got == 100:
                break
            s = ss[si]
            words = [m for m in WORD.finditer(s)
                     if sum(c in VOW for c in m.group().lower()) >= 2 and "ё" not in m.group().lower()
                     and (m.group()[0].islower() or m.start() == len(s) - len(s.lstrip()))]
            if not words:
                continue
            m = rnd.choice(words)
            if (m.group().lower(), s) in used:
                continue
            used.add((m.group().lower(), s))
            items.append({"книга": bi + 1, "фраза": s, "слово": m.group(), "позиция": m.start(),
                          "до": ss[si - 1] if si else (paras[pi - 1] if pi else ""),
                          "после": ss[si + 1] if si + 1 < len(ss) else "",
                          "абзац": paras[pi]})
            got += 1
    _os.makedirs(OUT, exist_ok=True)
    json.dump(items, open(f"{OUT}/книги_300.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"слов: {len(items)}")
    return 0


def zal_index(path: str) -> dict[str, set[int]]:
    """Форма без ударения -> номера ударных гласных (основное ударение U+0301; «ё»)."""
    out: dict[str, set[int]] = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 2 or p[0] == "word_without_stress_markers":
            continue
        w, st = p[0].lower(), p[1].lower()
        vi, idx = -1, None
        for i, c in enumerate(st):
            if c in VOW:
                vi += 1
                if c == "ё" or (i + 1 < len(st) and st[i + 1] == "́"):
                    idx = vi
        if idx is not None:
            out.setdefault(w.replace("ё", "е"), set()).add(idx)
    return out


def truth(zpath: str) -> int:
    """Верное ударение: Зализняк (однозначно) или ручной разбор."""
    items = json.load(open(f"{OUT}/книги_300.json", encoding="utf-8"))
    z = zal_index(zpath)
    manual = {}
    mp = f"{OUT}/разбор.tsv"
    if _os.path.exists(mp):
        for line in open(mp, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0].isdigit() and p[2].strip():
                manual[int(p[0])] = p[2].strip()
    need = []
    for i, it in enumerate(items):
        k = it["слово"].lower()
        zv = z.get(k)
        it["зализняк"] = sorted(zv) if zv else None
        if i in manual:
            it["верно"], it["источник"] = manual[i], "разбор"
        elif zv and len(zv) == 1:
            it["верно"], it["источник"] = _mark(k, next(iter(zv))), "Зализняк"
        else:
            it["верно"], it["источник"] = None, None
            need.append(i)
    json.dump(items, open(f"{OUT}/книги_300.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"по Зализняку: {sum(it['источник'] == 'Зализняк' for it in items)}, "
          f"разбор: {sum(it['источник'] == 'разбор' for it in items)}, без ответа: {len(need)}")
    return 0


def _mark(w: str, idx: int) -> str:
    out, vi = [], -1
    for c in w:
        if c in VOW:
            vi += 1
            if vi == idx:
                out.append("+")
        out.append(c)
    return "".join(out)


if __name__ == "__main__":
    cmd = _sys.argv[1] if len(_sys.argv) > 1 else "build"
    rc = build() if cmd == "build" else truth(_sys.argv[2])
    _sys.stdout.flush()
    _os._exit(rc)
