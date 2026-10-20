#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, п. 6: что дал каждый шаг разметки - по тексту (без синтеза).

Наборы: 50 трудных слов (ударения_50_39.txt, по 3 фразы в абзац, как в замере 11),
300 случайных слов из трёх книг (Архив/ударения 11а/книги_300.json, абзац целиком -
как его видит программа), три ошибки длинного куска 3.9.
Шаги (накопительно):
  0  как было (3.9): глава одним куском в RUAccent, silero при неуверенности
  1  по одному предложению (п. 1)
  1б + соседние предложения как контекст омографа (п. 1)
  2  омографы - решение RUAccent по предложению, без подмены silero (п. 2)
  2м то же + языковая модель на сомнительных (для сравнения, в программу не идёт)
  3  2 + справочник для споров на обычных словах (п. 3): Зализняк, затем Викисловарь
  3б 3 + Зализняк и без спора: обычное слово, где RUAccent поставил не то ударение
  итог - настройки программы по умолчанию (без соседних предложений: они испортили «Связи»)
Верно - ударная гласная как в эталоне; односложные - по буквам (все / всё).
Отдельно - сверка со словарём по Зализняку (форма, известная однозначно).

    python "Инструменты разработки/stress_steps_11a.py" [модель-судья [шаги…]]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import re
import time

OUT = "Архив/ударения 11а"
VOW = "аеёиоуыэюя"
LONG = "Архив/голоса диктора 3.9/длинный кусок/Глава первая (отрывок).txt"
LONG_WORDS = [("Связи требует", "Связи", "св+язи"), ("Малому Доминиону", "Малому", "м+алому"),
              ("Малого Доминиона Чиалори.", "Малого", "м+алого")]


def key(w):
    return w.replace("+", "").lower().replace("ё", "е")


def ok(got: str | None, truth: str) -> bool:
    from core.stressdict import stress_of
    if got is None:
        return False
    g, t = got.lower(), truth.lower()
    if sum(c in VOW for c in key(t)) == 1:
        return g.replace("+", "") == t.replace("+", "")
    return stress_of(g) == stress_of(t)


def token_at(marked: str, src: str, pos: int, word: str) -> str | None:
    """Слово исходного текста на позиции pos - в размеченном тексте."""
    j, i = 0, 0
    pm = {}
    while i < len(src) and j < len(marked):
        if marked[j] == "+":
            j += 1
            continue
        pm[i] = j
        i += 1
        j += 1
    if pos in pm and key(marked[pm[pos]:pm[pos] + len(word) + 4]).startswith(key(word)):
        j = pm[pos] - 1 if pm[pos] and marked[pm[pos] - 1] == "+" else pm[pos]   # «+» перед первой буквой
        m = re.match(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*", marked[j:])
        if m and key(m.group()) == key(word):
            return m.group()
    best = None                                  # буквы разошлись - ближайшее такое слово
    for m in re.finditer(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*", marked):
        if key(m.group()) == key(word) and (best is None or abs(m.start() - pos) < abs(best.start() - pos)):
            best = m
    return best.group() if best else None


def configure(step: str, S, model: str | None):
    if step == "итог":                       # как в программе по умолчанию
        import importlib
        importlib.reload(S)
        return
    S.BY_SENTENCE = step != "0"
    S.NEIGHBORS = step not in ("0", "1")
    # омографы: в 3.7 - silero, когда RUAccent не уверен (ниже 0,8); с шага 2 - решение
    # RUAccent по предложению без подмены (замер судей 11а: 85 из 90 против 81)
    S.SILERO_BELOW = 0.8 if step in ("0", "1", "1б") else 0.0
    S.JUDGE = step == "2м"
    S.REFERENCE = step in ("3", "3б")
    S.REFERENCE_ALL = step == "3б"
    S._judge_model[:] = [model] if S.JUDGE else []


def main() -> int:
    import core.pipeline as P
    from core import stress as S
    model = _sys.argv[1] if len(_sys.argv) > 1 else "qwen3:14b"
    rows = [l.rstrip("\n").split("|") for l in open("Инструменты разработки/ударения_50_39.txt", encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    rows = [(p.strip(), w.strip(), t.strip()) for p, w, t in rows]
    chunks: list[list] = []
    for r in rows:
        for ch in chunks:
            if len(ch) < 3 and all(r[1].lower() != x[1].lower() for x in ch):
                ch.append(r)
                break
        else:
            chunks.append([r])
    items = [it for it in json.load(open(f"{OUT}/книги_300.json", encoding="utf-8")) if it.get("верно")]
    long_src = open(LONG, encoding="utf-8").read()
    zal = {}
    zp = f"{OUT}/зализняк.tsv"
    if _os.path.exists(zp):
        import importlib.util
        spec = importlib.util.spec_from_file_location("b300", "Инструменты разработки/book300_11a.py")
        b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b)
        zal = {k: v for k, v in b.zal_index(zp).items() if len(v) == 1}
    res, detail = {}, {}
    steps = [a for a in _sys.argv[2:]] or ["0", "1", "1б", "2", "2м", "3", "3б"]
    for step in steps:
        configure(step, S, model)
        S.JUDGED.clear()
        for k in S.JUDGE_STATS:
            S.JUDGE_STATS[k] = 0
        t0 = time.time()
        r50, bad50 = 0, []
        for ch in chunks:
            mt = P._prepare_text(" ".join(p for p, _w, _t in ch), "+", True, True, "ru")
            off = 0
            src = " ".join(p for p, _w, _t in ch)
            for p, w, t in ch:
                start = src.index(p, off)
                pos = start + re.search(r"(?<![А-Яа-яЁё])" + re.escape(w), p, re.IGNORECASE).start()
                off = start + len(p)
                got = token_at(mt, src, pos, w)
                if ok(got, t):
                    r50 += 1
                else:
                    bad50.append(f"{w}: {got} (верно {t})")
        r300, bad300, zok, zn = 0, [], 0, 0
        cache = {}
        for it in items:
            para = it["абзац"]
            if para not in cache:
                cache[para] = P._prepare_text(para, "+", True, True, "ru")
            pos = para.index(it["фраза"]) + it["позиция"]
            got = token_at(cache[para], para, pos, it["слово"])
            good = ok(got, it["верно"])
            r300 += good
            if not good:
                bad300.append(f"{it['слово']}: {got} (верно {it['верно']})")
            zk = key(it["слово"])
            if zk in zal and got:
                from core.stressdict import stress_of
                zn += 1
                zok += stress_of(got.lower()) == (next(iter(zal[zk])),)
        lm = P._prepare_text(long_src, "+", True, True, "ru")
        rl, badl = 0, []
        for ctx, w, t in LONG_WORDS:
            pos = long_src.index(ctx) + ctx.index(w)
            got = token_at(lm, long_src, pos, w)
            if ok(got, t):
                rl += 1
            else:
                badl.append(f"{w}: {got}")
        res[step] = {"50 слов": r50, "300 слов": f"{r300} из {len(items)}",
                     "Зализняк": f"{zok} из {zn}", "длинный кусок": f"{rl} из 3",
                     "судья": S.JUDGE_STATS["судья"], "справочник": S.JUDGE_STATS["справочник"],
                     "сомнительных": S.JUDGE_STATS["сомнительных"], "с": round(time.time() - t0)}
        detail[step] = {"50": bad50, "300": bad300, "длинный": badl}
        print(step, json.dumps(res[step], ensure_ascii=False), flush=True)
        print("   ошибки 50:", bad50, flush=True)
        print("   ошибки 300:", bad300, flush=True)
    try:
        from core.homograph_llm import unload
        unload(model)
    except Exception:
        pass
    json.dump({"итог": res, "ошибки": detail}, open(f"{OUT}/шаги по тексту.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
