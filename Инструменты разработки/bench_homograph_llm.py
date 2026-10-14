#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Языковая модель как третий судья в омографах: даёт ли она прибавку.

Тот же открытый набор, что в bench_stress.py (примеры Викисловаря с ударением
во всех словах), - берутся омографы, у которых RUAccent знает варианты.
Сравниваются:
  * RUAccent; silero-stress; гибрид (как сейчас в программе);
  * модель решает КАЖДЫЙ омограф (выборка);
  * модель решает только там, где RUAccent и silero разошлись.
Плюс 30 фраз из слепого набора (эталон исправлен, см. docs/Приёмка 3.0).

    python "Инструменты разработки/bench_homograph_llm.py" --cache joint_cache.json \\
        --data ru_wikt.jsonl.gz --phrases 30_фраз_эталон.json [--model qwen3.5:9b]
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--all-sample", type=int, default=600)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import bench_stress as B
    from core.stress import _get_accentizer
    from core import homograph_llm as L
    ru = _get_accentizer()
    model = a.model or L.available_model()
    print("модель:", model, flush=True)
    outs = json.load(open(a.cache, encoding="utf-8"))
    cases = []           # (предложение без ударений, варианты, gold, ru, si)
    for s, o in outs.items():
        if not o:
            continue
        gw, rw, sw = B.WORD.findall(s), B.WORD.findall(o[0]), B.WORD.findall(o[1])
        if not (len(gw) == len(rw) == len(sw)):
            continue
        src = B.plain(s)
        for g, r, si in zip(gw, rw, sw):
            k = B.key(g)
            vs = ru.omographs.get(k)
            if not vs or B.nvow(g) < 2:
                continue
            gp = B.gold_pos(g)
            if gp is None:
                continue
            idx = [B.pred_pos(v) for v in vs]
            if gp not in idx:
                continue
            cases.append((src, vs, idx.index(gp), B.pred_pos(r), B.pred_pos(si), idx))
    print("омографов с вариантами:", len(cases), flush=True)

    def acc(pred):
        return round(100 * sum(1 for c, p in zip(cases, pred) if p == c[5][c[2]]) / len(cases), 2)

    ru_p = [c[3] for c in cases]
    si_p = [c[4] for c in cases]
    res = {"омографов": len(cases), "RUAccent": acc(ru_p), "silero": acc(si_p), "гибрид": acc(si_p)}
    dis = [i for i, c in enumerate(cases) if c[3] != c[4]]
    res["расходятся"] = len(dis)
    # модель только на расхождениях
    t0 = time.time()
    s3 = list(si_p)
    llm_cache = {}
    for n, i in enumerate(dis, 1):
        c = cases[i]
        ch = L.choose(c[0], c[1], model)
        llm_cache[i] = ch
        if ch is not None:
            s3[i] = c[5][ch]
        if n % 50 == 0:
            print(f"  расхождения: {n}/{len(dis)} за {time.time() - t0:.0f} с", flush=True)
    res["модель на расхождениях"] = acc(s3)
    res["время на расхождение, с"] = round((time.time() - t0) / max(1, len(dis)), 2)
    # модель на всех (выборка)
    random.seed(3)
    samp = random.sample(range(len(cases)), min(a.all_sample, len(cases)))
    ok_llm = ok_h = 0
    t0 = time.time()
    for i in samp:
        c = cases[i]
        ch = llm_cache.get(i) if i in llm_cache else L.choose(c[0], c[1], model)
        p = c[5][ch] if ch is not None else si_p[i]
        ok_llm += p == c[5][c[2]]
        ok_h += si_p[i] == c[5][c[2]]
    res["выборка «модель на всех»"] = len(samp)
    res["модель на всех (выборка)"] = round(100 * ok_llm / len(samp), 2)
    res["гибрид на той же выборке"] = round(100 * ok_h / len(samp), 2)
    res["время на омограф, с"] = round((time.time() - t0) / len(samp), 2)
    # 30 фраз
    from core.normalize import normalize_text
    from core.stress import add_stress, add_stress_text
    ph = json.load(open(a.phrases, encoding="utf-8"))
    pr = {"RUAccent": 0, "гибрид": 0, "модель на расхождениях": 0, "модель на всех": 0}
    for text, gold in ph:
        k = gold.replace("+", "").lower().replace("ё", "е")
        n = normalize_text(text, language="ru")
        def mark(t):
            for w in re.findall(r"[А-Яа-яЁё+]+", t):
                if w.replace("+", "").lower().replace("ё", "е") == k:
                    return B.pred_pos(w)
        gp = B.pred_pos(gold)
        r_p, h_p = mark(add_stress(n)), mark(add_stress_text(n))
        vs = ru.omographs.get(k)
        l_p = None
        if vs:
            ch = L.choose(n, vs, model)
            l_p = B.pred_pos(vs[ch]) if ch is not None else None
        pr["RUAccent"] += r_p == gp
        pr["гибрид"] += h_p == gp
        pr["модель на всех"] += (l_p if l_p is not None else h_p) == gp
        pr["модель на расхождениях"] += ((l_p if (l_p is not None and r_p != h_p) else h_p) == gp)
    res["30 фраз"] = pr
    L.unload(model)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
