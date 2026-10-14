#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Разбор ударений на настоящей книге: где ошибка вероятнее всего.

Ошибка в ударении бывает двух родов, и лечатся они по-разному:
  (а) неверный «+» в тексте — ошибся расстановщик;
  (б) «+» верный, но модель синтеза его не послушалась.

Этот разбор отвечает на (а), насколько это можно без ручной разметки книги:
  * сколько слов вне словаря (имена, выдуманные слова) — там расстановщик
    угадывает по правилам, и ошибки там вероятнее всего;
  * где два независимых расстановщика (RUAccent и silero-stress) расходятся —
    это и есть кандидаты в ошибки (а), их слушает человек;
  * одинаково ли ставится ударение у одного и того же слова вне словаря в
    разных местах книги.

(б) отсюда не видно — это проверяется на слух короткими фразами.

    python "Инструменты разработки/analyze_book_stress.py" КНИГА.fb2 --parts 1 7 13 --out ПАПКА
"""
from __future__ import annotations

import argparse, difflib, json, os, re, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
WORD = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")


def nvow(w): return sum(c in VOWELS for c in w)
def plain(w): return w.replace("+", "")
def key(w): return plain(w).lower().replace("ё", "е")


def spos(w):
    vi, plus, yo = -1, False, None
    for c in w:
        if c == "+":
            plus = True
            continue
        if c in VOWELS:
            vi += 1
            if plus:
                return vi
            if c in "ёЁ" and yo is None:
                yo = vi
        plus = False
    return yo


def mark(w, p):
    """Слово с ударением на p-й гласной — для отчёта, заглавной буквой."""
    if p is None:
        return plain(w).lower()
    out, vi = [], -1
    for c in plain(w).lower():
        if c in VOWELS:
            vi += 1
            out.append(c.upper() if vi == p else c)
        else:
            out.append(c)
    return "".join(out)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--parts", type=int, nargs="+", default=[1, 7, 13])
    ap.add_argument("--out", required=True)
    ap.add_argument("--whole-book-oov", action="store_true",
                    help="проверить единообразие слов вне словаря по всей книге")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from core.readers import read_document
    from core.chapters import build_chapters
    from core.normalize import normalize_text
    from core.stress import _get_accentizer
    from silero_stress import load_accentor

    doc = read_document(a.book)
    chs = build_chapters(doc, a.minutes)
    ru = _get_accentizer()
    si = load_accentor()
    accents, omographs = ru.accents, ru.omographs

    def in_dict(k):
        return k in accents or k in omographs

    stats = Counter()
    disagree = Counter()          # (ключ, вариант RU, вариант silero, род) -> раз
    examples = {}
    t_ru = t_si = 0.0
    for n in a.parts:
        if n > len(chs):
            continue
        text = normalize_text(chs[n - 1].text, language="ru")
        t0 = time.time(); r_out = ru.process_all(text); t_ru += time.time() - t0
        t0 = time.time(); s_out = " ".join(si(s) for s in sentences(text)); t_si += time.time() - t0
        open(os.path.join(a.out, f"Часть {n:02d} — RUAccent.txt"), "w", encoding="utf-8").write(r_out)
        open(os.path.join(a.out, f"Часть {n:02d} — silero-stress.txt"), "w", encoding="utf-8").write(s_out)
        rw, sw = WORD.findall(r_out), WORD.findall(s_out)
        sm = difflib.SequenceMatcher(None, [key(w) for w in rw], [key(w) for w in sw], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                stats["слова не сопоставились"] += i2 - i1
                continue
            for r, s in zip(rw[i1:i2], sw[j1:j2]):
                if nvow(r) < 2:
                    continue
                k = key(r)
                kind = "омограф" if k in omographs else ("словарное" if k in accents else "вне словаря")
                stats[f"слов: {kind}"] += 1
                stats["слов всего"] += 1
                pr, ps = spos(r), spos(s)
                if pr is None:
                    stats[f"RUAccent без ударения: {kind}"] += 1
                if pr != ps:
                    stats[f"расхождений: {kind}"] += 1
                    d = (k, mark(r, pr), mark(s, ps), kind)
                    disagree[d] += 1
                    examples.setdefault(d, s)
    stats["время RUAccent, с"] = round(t_ru, 1)
    stats["время silero-stress, с"] = round(t_si, 1)

    # единообразие слов вне словаря по всей книге
    oov_var = {}
    if a.whole_book_oov:
        var = defaultdict(Counter)
        for ch in chs:
            out = ru.process_all(normalize_text(ch.text, language="ru"))
            for w in WORD.findall(out):
                k = key(w)
                if nvow(w) > 1 and not in_dict(k):
                    var[k][spos(w)] += 1
        multi = {k: v for k, v in var.items() if len(v) > 1}
        stats["вне словаря: разных слов по книге"] = len(var)
        stats["вне словаря: употреблений по книге"] = sum(sum(v.values()) for v in var.values())
        stats["вне словаря: слов с разным ударением"] = len(multi)
        stats["вне словаря: употреблений у таких слов"] = sum(sum(v.values()) for v in multi.values())
        oov_var = {k: {mark(k, p): n for p, n in v.most_common()}
                   for k, v in sorted(multi.items(), key=lambda kv: -sum(kv[1].values()))}

    res = {"статистика": dict(stats),
           "расхождения": [{"слово": k, "RUAccent": r, "silero": s, "род": kind, "раз": n}
                           for (k, r, s, kind), n in disagree.most_common()],
           "вне словаря, разное ударение по книге": oov_var}
    json.dump(res, open(os.path.join(a.out, "разбор.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  разных расхождений: {len(disagree)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
