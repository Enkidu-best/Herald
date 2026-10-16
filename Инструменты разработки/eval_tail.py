#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Растянутый конец куска («никогооо», 3.4 п. 7): длина хвоста последнего слова.

Текст главы выравнивается по звуку целиком (core/align). У каждого слова в
конце предложения меряется «хвост» - сколько ещё звучит голос после
выровненного конца слова (по громкости, до тишины). Слова делятся на два
рода: последнее слово КУСКА (там модель упирается в конец окна) и конец
предложения ВНУТРИ куска - это мерка нормы для того же голоса.

Ещё печатает, что акустическая модель слышит на первых словах («Логично»).

    python "Инструменты разработки/eval_tail.py" --text отрывок.txt \\
        --wav "3.2=глава_нет.wav:280" --wav "3.3=глава_звук_187.wav:187"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json

import numpy as np


def tail_after(audio, sr, t_end, frame=0.01, quiet_db=-35.0, hold=0.08, limit=2.0):
    """Сколько секунд после t_end ещё звучит голос (до тишины не короче hold)."""
    w = int(frame * sr)
    a = audio[int(t_end * sr): int((t_end + limit + hold) * sr)]
    n = len(a) // w
    if n == 0:
        return 0.0
    e = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    ref = np.percentile(np.sqrt(np.mean(audio[:len(audio) // w * w].reshape(-1, w) ** 2,
                                        axis=1) + 1e-12), 95)
    loud = 20 * np.log10(e / ref) > quiet_db
    need = int(hold / frame)
    run = 0
    for i, on in enumerate(loud):
        run = 0 if on else run + 1
        if run >= need:
            return max(0.0, (i - need + 1) * frame)
    return limit


def voiced_between(audio, sr, t0, t1, frame=0.01, quiet_db=-35.0):
    """Сколько секунд между t0 и t1 звучит голос (громкие кадры), а не тишина."""
    w = int(frame * sr)
    a = audio[int(t0 * sr): int(t1 * sr)]
    n = len(a) // w
    if n <= 0:
        return 0.0
    allf = audio[:len(audio) // w * w].reshape(-1, w)
    ref = np.percentile(np.sqrt(np.mean(allf ** 2, axis=1) + 1e-12), 95)
    e = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    return float(np.sum(20 * np.log10(e / ref) > quiet_db) * frame)


VOW = set("аеёиоуыэюя")


def stress_marks(text_plus: str, vocab) -> list:
    """Для каждого слова (как у выравнивателя) - номер ударной гласной или None."""
    import re
    out = []
    for m in re.finditer(r"[\w\-+]+", text_plus):
        for part in m.group(0).split("-"):
            clean = part.replace("+", "").lower()
            if not any(c in vocab and c != "|" for c in clean.replace("ё", "е")):
                continue
            idx, vi = None, 0
            for c in part:
                if c == "+":
                    idx = vi
                    continue
                if c.lower() in VOW:
                    if c.lower() == "ё" and idx is None:
                        idx = vi
                    vi += 1
            out.append(idx)
    return out


def vowel_prominence(audio, sr, w, frame=0.01):
    """[(гласная, длительность, громкость)] по пикам букв (до пика следующей)."""
    sp = w.spans
    out = []
    for k, (ch, t0, _t1) in enumerate(sp):
        if ch not in VOW and ch != "е":
            continue
        t_next = sp[k + 1][1] if k + 1 < len(sp) else min(w.end + 0.25, t0 + 0.4)
        d = max(0.02, t_next - t0)
        seg = audio[int(t0 * sr): int((t0 + d) * sr)]
        e = float(np.sqrt(np.mean(seg ** 2) + 1e-12)) if len(seg) else 0.0
        out.append((ch, d, e))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--wav", action="append", required=True, metavar="ИМЯ=ФАЙЛ:ДЛИНА")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import soundfile as sf
    import core.pipeline as P
    from core import align
    al = align.aligner().load()
    src = open(a.text, encoding="utf-8").read()
    prepared = P._prepare_text(src, "+", True, True, "ru")
    res = {}
    for spec in a.wav:
        name, rest = spec.split("=", 1)
        path, mc = rest.rsplit(":", 1)
        audio, sr = sf.read(path, dtype="float32")
        chunks = P._chunk_paragraphs(prepared, int(mc), 0.4, 0.8, 1.1)
        # номер последнего слова каждого куска (по словам выравнивателя)
        ends, k = set(), -1
        for c in chunks:
            k += len(al._words(P._with_end_punct(c[0]).replace("+", "")))
            ends.add(k)
        text = " ".join(P._with_end_punct(c[0]) for c in chunks).replace("+", "")
        ws = al.score(audio, sr, text)
        # «звучит после слова»: голос между выровненным концом слова (пик последней
        # буквы) и началом следующего - это растянутая последняя гласная
        vb = []
        for i, w in enumerate(ws[:-1]):
            v = voiced_between(audio, sr, w.end, ws[i + 1].start)
            vb.append({"i": i, "слово": w.word, "после": round(v, 3), "t": round(w.end, 2),
                       "знак": w.punct_after, "конец_куска": i in ends})
        vals = np.array([r["после"] for r in vb])
        print(f"\n{name}: звучит после слова - медиана {np.median(vals):.2f} с, "
              f"99-й процентиль {np.percentile(vals, 99):.2f}, наибольшие:")
        for r in sorted(vb, key=lambda r: -r["после"])[:8]:
            m, s_ = divmod(r["t"], 60)
            print(f"    {int(m)}:{s_:04.1f} «{r['слово']}{r['знак']}» {r['после']:.2f} с"
                  f"{' (конец куска)' if r['конец_куска'] else ''}")
        for r in vb:
            if r["слово"].lower().startswith("никог"):
                print(f"  «никого»: звучит после {r['после']:.2f} с")
        for kind in (",", "."):
            v = [r["после"] for r in vb if r["знак"] == kind]
            if v:
                print(f"  перед «{kind}»: медиана {np.median(v):.2f}, 95-й {np.percentile(v, 95):.2f}")
        # ударение: какая гласная звучит сильнее всех (длина × громкость) - та ли,
        # что помечена «+»
        marks = stress_marks(" ".join(P._with_end_punct(c[0]) for c in chunks), al._vocab)
        firsts, k = set(), 0
        for c in chunks:
            firsts.add(k)
            k += len(al._words(P._with_end_punct(c[0]).replace("+", "")))
        st_rows = []
        if len(marks) == len(ws):
            for i, (w, idx) in enumerate(zip(ws, marks)):
                vp = vowel_prominence(audio, sr, w)
                if idx is None or len(vp) < 2 or idx >= len(vp):
                    continue
                score = [d * e for _c, d, e in vp]
                top = int(np.argmax(score))
                ratio = score[idx] / max(score)
                st_rows.append({"i": i, "слово": w.word, "ударная": idx, "сильнее": top,
                                "доля": round(ratio, 2), "первое_в_куске": i in firsts,
                                "t": round(w.start, 2)})
            ok = [r for r in st_rows if r["ударная"] == r["сильнее"]]
            fi = [r for r in st_rows if r["первое_в_куске"]]
            print(f"  ударение по звуку совпало с «+»: {len(ok)} из {len(st_rows)} слов; "
                  f"первые слова кусков: {sum(r['ударная'] == r['сильнее'] for r in fi)} из {len(fi)}")
            for r in st_rows[:3] + [r for r in st_rows if r["слово"].lower() == "логично"]:
                print(f"    «{r['слово']}» {r['t']:.1f} с: ударная №{r['ударная'] + 1}, "
                      f"сильнее №{r['сильнее'] + 1}, доля ударной {r['доля']}")
        else:
            print(f"  ! слов с ударением {len(marks)}, у выравнивателя {len(ws)} - не сверить")
        res[name + " ударение"] = st_rows
        rows = []
        for i, w in enumerate(ws):
            if not (w.punct_after in (".", "!", "?", "…", ":", ";") or i in ends):
                continue
            t = tail_after(audio, sr, w.end)
            rows.append({"i": i, "слово": w.word, "конец_куска": i in ends,
                         "хвост": round(t, 3), "t": round(w.end, 2), "слышно": w.heard})
        res[name] = rows
        ce = [r["хвост"] for r in rows if r["конец_куска"]]
        ie = [r["хвост"] for r in rows if not r["конец_куска"]]
        print(f"\n{name}: кусков {len(chunks)}")
        for title, v in (("конец куска", ce), ("конец предложения внутри куска", ie)):
            if v:
                print(f"  {title}: {len(v)} слов, хвост медиана {np.median(v):.2f} с, "
                      f"90-й процентиль {np.percentile(v, 90):.2f}, наибольший {max(v):.2f}")
        for r in sorted(rows, key=lambda r: -r["хвост"])[:8]:
            m, s = divmod(r["t"], 60)
            print(f"    {int(m)}:{s:04.1f} «{r['слово']}» хвост {r['хвост']:.2f} с"
                  f"{' (конец куска)' if r['конец_куска'] else ''}")
        print("  первые слова:", ", ".join(f"«{w.word}» слышно «{w.heard}» ({w.post_mean:.1f})"
                                          for w in ws[:4]))
        for w in ws:
            if w.word.lower().startswith("никог"):
                print(f"  «{w.word}» {w.end:.1f} с: хвост {tail_after(audio, sr, w.end):.2f} с")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
