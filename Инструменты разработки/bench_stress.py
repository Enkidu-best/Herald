#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение расстановщиков ударений на ОТКРЫТОЙ разметке.

Тестовые наборы авторов закрыты: у RUAccent — 2 млн собственных предложений из
аудиоразметки, у silero-stress — валидация, описанная в вики, но не выложенная.
Поэтому берём то, что размечено людьми и открыто: русские статьи английского
Викисловаря (выгрузка kaikki.org, CC-BY-SA). В них:

  * примеры-предложения, где ударение проставлено во всех словах — это проверка
    в контексте, включая омографы;
  * таблицы словоформ с ударением — проверка «словарных форм».

Омограф — форма, у которой в Викисловаре встречается больше одного места
ударения (за́мок / замо́к, сте́ны / стены́).

Оговорка: оба расстановщика могли учиться на словарях, куда вошёл и Викисловарь,
поэтому абсолютные числа завышены. Сравнение между ними от этого честнее не
становится хуже: условия одинаковые.

    python "Инструменты разработки/bench_stress.py" ru_wikt.jsonl.gz --n 4000
"""
from __future__ import annotations

import argparse, gzip, json, random, re, sys, time
from collections import Counter, defaultdict

ACUTE = "́"
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
WORD = re.compile(r"[А-Яа-яЁё́+]+(?:-[А-Яа-яЁё́+]+)*")


def plain(w: str) -> str:
    return w.replace(ACUTE, "").replace("+", "")


def gold_pos(w: str) -> int | None:
    """Номер ударной гласной (среди гласных) по разметке с акутом."""
    vi = -1
    for i, c in enumerate(w):
        if c in VOWELS:
            vi += 1
            if i + 1 < len(w) and w[i + 1] == ACUTE:
                return vi
    return None


def pred_pos(w: str) -> int | None:
    """Номер ударной гласной по выводу с «+» (или ё без плюса — она ударна)."""
    vi = -1
    plus = False
    yo = None
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


def nvow(w: str) -> int:
    return sum(c in VOWELS for c in w)


def key(w: str) -> str:
    return plain(w).lower().replace("ё", "е")


def load(path: str):
    stress_by_form: dict[str, set] = defaultdict(set)
    sents, forms = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for fm in d.get("forms", []):
                t = fm.get("form", "")
                if " " in t or not re.fullmatch(r"[А-Яа-яЁё́-]+", t or ""):
                    continue
                gp = gold_pos(t) if ACUTE in t else None
                if gp is None and "ё" in t.lower():
                    gp = pred_pos(t)
                if gp is not None and nvow(t) > 1:
                    stress_by_form[key(t)].add(gp)
                    forms.append(t)
            for s in d.get("senses", []):
                for e in s.get("examples", []):
                    t = (e.get("text") or "").strip()
                    if ACUTE in t and len(WORD.findall(t)) >= 4 and len(t) < 300:
                        sents.append(t)
    homos = {k for k, v in stress_by_form.items() if len(v) > 1}
    return sents, forms, homos


def run_ruaccent():
    # берём ровно тот RUAccent, что в программе, — с починкой входов ONNX
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
        __import__("os").path.abspath(__file__))))
    from core.stress import _get_accentizer
    return _get_accentizer().process_all


def run_silero():
    from silero_stress import load_accentor
    a = load_accentor()
    return lambda s: a(s)


def score(fn, items, homos):
    """items: [(предложение с акутами)] -> счётчики по словам."""
    c = Counter()
    misses = Counter()
    t0 = time.time()
    for s in items:
        src = plain(s)
        try:
            out = fn(src)
        except Exception:
            c["сбой"] += 1
            continue
        gw = WORD.findall(s)
        pw = WORD.findall(out)
        if len(gw) != len(pw):
            c["несовпадение разбиения"] += 1
            continue
        for g, p in zip(gw, pw):
            if nvow(g) < 2:
                continue
            gp = gold_pos(g)
            if gp is None and "ё" in g.lower():
                gp = pred_pos(g)
            if gp is None:
                continue                    # в разметке ударение не проставлено
            pp = pred_pos(p)
            h = key(g) in homos
            tag = "омограф" if h else "обычное"
            c[f"{tag}: всего"] += 1
            if pp is None:
                c[f"{tag}: без ударения"] += 1
            elif pp == gp:
                c[f"{tag}: верно"] += 1
            else:
                misses[(key(g), h)] += 1
    c["время, с"] = round(time.time() - t0, 1)
    return c, misses


def joint(test, homos, ru_fn, si_fn, ru_known, cache=None):
    """Оба расстановщика на ОДНИХ И ТЕХ ЖЕ предложениях, плюс гибрид.

    Гибрид: RUAccent везде (его разметку видела модель синтеза при дообучении),
    но в омографах — выбор silero-stress. Считаем только предложения, где у
    обоих совпало разбиение на слова, — иначе сравнение нечестное.
    """
    outs = {}
    if cache and __import__("os").path.exists(cache):
        outs = json.load(open(cache, encoding="utf-8"))
    for s in test:
        if s in outs:
            continue
        src = plain(s)
        try:
            outs[s] = [ru_fn(src), si_fn(src)]
        except Exception:
            outs[s] = None
    if cache:
        json.dump(outs, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    c = Counter()
    for s in test:
        o = outs.get(s)
        if not o:
            c["сбой"] += 1
            continue
        gw, rw, sw = WORD.findall(s), WORD.findall(o[0]), WORD.findall(o[1])
        if not (len(gw) == len(rw) == len(sw)):
            c["пропущено предложений (разбиение)"] += 1
            continue
        c["предложений"] += 1
        for g, r, si in zip(gw, rw, sw):
            if nvow(g) < 2:
                continue
            gp = gold_pos(g)
            if gp is None and "ё" in g.lower():
                gp = pred_pos(g)
            if gp is None:
                continue
            k = key(g)
            tag = "омограф" if k in homos else "обычное"
            c[f"{tag}: всего"] += 1
            pr, ps = pred_pos(r), pred_pos(si)
            ph = ps if (k in homos or not ru_known(k)) else pr
            for name, pp in (("RUAccent", pr), ("silero", ps), ("гибрид", ph)):
                if pp is None:
                    c[f"{tag}: {name} без ударения"] += 1
                elif pp == gp:
                    c[f"{tag}: {name} верно"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--n", type=int, default=4000, help="предложений без омографов")
    ap.add_argument("--forms", type=int, default=5000)
    ap.add_argument("--json")
    ap.add_argument("--joint", help="кэш выводов; сравнение на общем наборе + гибрид")
    a = ap.parse_args()
    if a.joint:
        sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
            __import__("os").path.abspath(__file__))))
    random.seed(7)
    sents, forms, homos = load(a.data)
    with_h = [s for s in sents if any(key(w) in homos and nvow(w) > 1 for w in WORD.findall(s))]
    without = [s for s in sents if s not in set(with_h)]
    test = with_h + random.sample(without, min(a.n, len(without)))
    uniq_forms = sorted({f for f in forms if key(f) not in homos})
    fsample = random.sample(uniq_forms, min(a.forms, len(uniq_forms)))
    print(f"предложений с ударениями: {len(sents)}, с омографами: {len(with_h)}, "
          f"в тесте: {len(test)}; омографов-форм: {len(homos)}; "
          f"словоформ для проверки: {len(fsample)}", file=sys.stderr)
    if a.joint:
        from core.stress import _get_accentizer
        ru = _get_accentizer()
        c = joint(test, homos, ru.process_all, run_silero(),
                  lambda k: k in ru.accents or k in ru.omographs, cache=a.joint)
        print(json.dumps(c, ensure_ascii=False, indent=1))
        return 0
    res = {}
    for name, mk in (("RUAccent turbo3.1", run_ruaccent), ("silero-stress 1.5", run_silero)):
        fn = mk()
        c, miss = score(fn, test, homos)
        # словоформы — по одной, без контекста
        fc = Counter()
        t0 = time.time()
        for w in fsample:
            out = fn(plain(w))
            pw = WORD.findall(out)
            fc["всего"] += 1
            gp = gold_pos(w) if ACUTE in w else pred_pos(w)
            if pw and pred_pos(pw[0]) == gp:
                fc["верно"] += 1
        fc["время, с"] = round(time.time() - t0, 1)
        res[name] = {"контекст": dict(c), "словоформы": dict(fc),
                     "частые ошибки на омографах": [
                         (w, n) for (w, h), n in miss.most_common(200) if h][:25]}
        print(name, json.dumps(res[name], ensure_ascii=False), file=sys.stderr)
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
