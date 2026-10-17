#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Упрямые слова (3.8, п. 1): слушается ли клон «+», и какой приём помогает.

Этапы:
  silero  - проверка детектора ударения (core/stressdetect) на Silero: он «+»
            слушается строго, поэтому ударение на каждом слоге по очереди даёт
            готовую разметку. Печатает точность при разных весах мерок.
  f5      - клон читает набор слов (упрямые слова Андрея + кандидаты) каждым
            приёмом записи; детектор говорит, где ударение вышло. Звук - в
            --out-dir, итоги - в итоги.json.

    python "Инструменты разработки/bench_stubborn.py" silero
    python "Инструменты разработки/bench_stubborn.py" f5 --out-dir Архив/упрямые
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import itertools
import json
import time

import numpy as np

CARRIER = "Скажи слово {} ещё раз."
# обычные слова 2-4 слогов - для проверки детектора (ударение ставим сами)
COMMON = """молоко собака дорога голова корова береза ворона машина картина работа
погода мужчина девочка комната улица история бумага подарок человек правило
остров озеро город берег парус мельница колокол пятница середина огород
мальчик облако яблоко зеркало дерево воробей телефон карандаш солдатик""".split()


def silero_stage(a) -> int:
    import soundfile as sf
    from core.engines import get_engine
    from core import align, stressdetect as SD
    eng = get_engine("silero")
    eng.ensure_loaded()
    al = align.aligner()
    rows = []
    t0 = time.time()
    for voice in a.voices.split(","):
        for w in COMMON:
            vpos = [i for i, c in enumerate(w) if c in SD.VOWELS]
            for k, i in enumerate(vpos):
                marked = w[:i] + "+" + w[i:]
                text = CARRIER.format(marked)
                audio = eng.synth_chunk(text, voice)
                sr = eng.sample_rate
                ws = [x for x in al.score(audio, sr, text) if x.word.lower() == w]
                if not ws:
                    continue
                vs = SD.vowels(audio, sr, ws[0])
                if len(vs) != len(vpos):
                    continue
                if a.save:                      # для разбора без модели
                    _os.makedirs(a.save, exist_ok=True)
                    name = f"{voice}__{w}__{k}"
                    a16 = align._resample(np.asarray(audio, np.float32), sr, 16000)
                    sf.write(_os.path.join(a.save, name + ".wav"), a16, 16000)
                    np.save(_os.path.join(a.save, name + ".npy"), al.emissions(a16).astype(np.float16))
                rows.append({"voice": voice, "word": w, "true": k, "text": text,
                             "spans": ws[0].spans, "start": ws[0].start, "end": ws[0].end,
                             "v": [[x.dur, x.en, x.f0] for x in vs]})
        print(f"{voice}: {len(rows)} вариантов, {time.time() - t0:.0f} с", flush=True)
    json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False)
    report(rows)
    return 0


def report(rows):
    from core import stressdetect as SD
    grid = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 0.5), (1, 0.5, 0.5),
            (1, 1, 1), (2, 1, 0.5), (1, 2, 0.5), (1, 0.5, 0), (0.5, 1, 0.5)]
    for w in grid:
        ok = n = 0
        for r in rows:
            vs = [SD.Vowel(0, 0, 0, d, e, f) for d, e, f in r["v"]]
            k, _m = SD.stressed(vs, w)
            ok += k == r["true"]
            n += 1
        print(f"  веса длит./громк./тон {w}: {ok}/{n} = {ok / max(n, 1):.0%}")


CARRIER_MARKED = "Скаж+и сл+ово {} ещё р+аз."
TRICKS = ["plus", "acute", "plus_acute", "reduce", "double", "hyphen", "upper", "reduce_double"]
# основной набор (после первых 7 слов Герасимова дефис и заглавная - заметно хуже)
MAIN_TRICKS = "plus,acute,reduce,double,reduce_double"


def trick(word: str, how: str) -> str:
    """Слово с «+» -> запись для модели этим приёмом."""
    i = word.index("+")
    v = word[i + 1]                                   # ударная гласная
    pre, post = word[:i], word[i + 2:]
    if how == "plus":
        return word
    if how == "acute":
        return pre + v + "\u0301" + post
    if how == "plus_acute":
        return pre + "+" + v + "\u0301" + post
    if how == "double":
        return pre + "+" + v + v + post
    if how == "upper":
        return pre + "+" + v.upper() + post
    if how == "hyphen":
        if post:
            return pre + "+" + v + "-" + post
        k = len(pre) - 1                              # «долг+и» -> «дол-г+и»
        return pre[:k] + "-" + pre[k:] + "+" + v
    if how == "reduce_double":
        r = trick(word, "reduce")
        j = r.index("+")
        return r[:j + 2] + r[j + 1] + r[j + 2:]
    if how == "reduce":                               # безударные о -> а, е -> и
        def red(part: str, first: bool) -> str:
            out = []
            for j, c in enumerate(part):
                prev = part[j - 1] if j else ("" if first else v)
                if c == "о":
                    c = "а"
                elif c == "е" and prev and prev not in SDV and prev not in "ьъ":
                    c = "и"
                out.append(c)
            return "".join(out)
        return red(pre, True) + "+" + v + red(post, False)
    raise ValueError(how)


SDV = set("аеёиоуыэюя")


def f5_stage(a) -> int:
    import mlx.core as mx
    import soundfile as sf
    from core import align, stressdetect as SD
    from core.engines import get_engine, clone_engine
    from core import profiles
    import core.pipeline as P
    words = [l.split("|") for l in open("Инструменты разработки/упрямые_слова.txt", encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    if a.only:
        words = [w for w in words if w[1].strip() in a.only.split(",")]
    # порядок: обычные (проверка детектора на клоне), слова Андрея, редкие
    order = {"обычное": 0, "Андрей": 1, "редкое": 2}
    words.sort(key=lambda w: order.get(w[1].strip(), 3))
    rare = [w for w in words if w[1].strip() == "редкое"][:a.limit_rare]
    words = [w for w in words if w[1].strip() != "редкое"] + rare
    tricks = a.tricks.split(",")
    al = align.aligner()
    al.load()
    blank = al._blank
    _os.makedirs(a.out_dir, exist_ok=True)

    def seg_of(audio, sr, plain):
        text = CARRIER.format(plain)
        lp = al.emissions(align._resample(np.asarray(audio, np.float32), sr, 16000))
        ws = [x for x in al.score(audio, sr, text) if x.word.lower() == plain.split("-")[0]]
        return SD.segment(lp, ws[0].spans, blank) if ws and ws[0].spans else None

    # образцы: Silero, ударение на каждом слоге, четыре голоса
    t0 = time.time()
    sil = get_engine("silero")
    sil.ensure_loaded()
    refs = {}
    for w, _src in words:
        plain = w.replace("+", "")
        vpos = [i for i, c in enumerate(plain) if c in SDV]
        refs[plain] = {}
        for k, i in enumerate(vpos):
            for voice in ("aidar", "eugene", "xenia", "kseniya"):
                au = sil.synth_chunk(CARRIER.format(plain[:i] + "+" + plain[i:]), voice)
                s = seg_of(au, sil.sample_rate, plain)
                if s is not None:
                    refs[plain].setdefault(k, []).append(s)
    print(f"образцы Silero: {time.time() - t0:.0f} с", flush=True)
    del sil
    eng = get_engine(clone_engine(), language="ru", **profiles.engine_kwargs(profiles.get(None), "balanced"))
    eng.ensure_loaded()
    rows = []
    for voice in a.voices.split(","):
        eng.synth_batch(["Пров+ерка."], voice)
        t0 = time.time()
        for wi, (w, src) in enumerate(words):
            plain, true = w.replace("+", ""), SD.vowel_index(w)
            for how in tricks:
                mx.random.seed(a.seed + wi)
                text = CARRIER_MARKED.format(trick(w, how))
                au = eng.synth_batch([text], voice)[0]
                sf.write(_os.path.join(a.out_dir, f"{voice}__{how}__{plain}.wav"), au, 24000)
                s = seg_of(au, 24000, plain)
                pred, margin = SD.by_reference(s, refs[plain]) if s is not None else (-1, 0.0)
                rows.append({"voice": voice, "trick": how, "word": w, "src": src.strip(),
                             "true": true, "pred": pred, "margin": round(margin, 4)})
            json.dump(rows, open(_os.path.join(a.out_dir, "итоги.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=0)
        print(f"{voice}: {time.time() - t0:.0f} с", flush=True)
    summary(rows)
    return 0


LETTERS = "БДН"


def blind_stage(a) -> int:
    """Слепой набор: на фразу три варианта - «+», «а/и», «+» с проверкой ударения
    по звуку (до 4 попыток с разным зерном, берётся та, где детектор слышит
    верное ударение). Буквы - случайно для каждого голоса; ключ - в docs."""
    import random
    import subprocess
    import mlx.core as mx
    import soundfile as sf
    from core import align, stressdetect as SD, profiles
    from core.engines import get_engine, clone_engine
    import core.pipeline as P
    rows = [l.rstrip("\n").split("|") for l in open("Инструменты разработки/ударения_38.txt",
                                                    encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    al = align.aligner()
    al.load()
    blank = al._blank

    def seg_in(audio, text, word):
        lp = al.emissions(align._resample(np.asarray(audio, np.float32), 24000, 16000))
        ws = [x for x in al.score(audio, 24000, text) if x.word.lower() == word]
        return SD.segment(lp, ws[0].spans, blank) if ws and ws[0].spans else None

    sil = get_engine("silero")
    sil.ensure_loaded()
    refs = {}
    for _ph, w in rows:
        plain = w.replace("+", "").lower()
        vpos = [i for i, c in enumerate(plain) if c in SDV]
        refs[plain] = {}
        for k, i in enumerate(vpos):
            for voice in ("aidar", "eugene", "xenia", "kseniya"):
                t = CARRIER.format(plain[:i] + "+" + plain[i:])
                au = sil.synth_chunk(t, voice)
                au = align._resample(np.asarray(au, np.float32), sil.sample_rate, 24000)
                sg = seg_in(au, CARRIER.format(plain), plain)
                if sg is not None:
                    refs[plain].setdefault(k, []).append(sg)
    del sil
    eng = get_engine(clone_engine(), language="ru",
                     **profiles.engine_kwargs(profiles.get(None), "balanced"))
    eng.ensure_loaded()
    key_lines = []
    for voice in a.voices.split(","):
        eng.synth_batch(["Пров+ерка."], voice)
        rnd = random.Random(f"3.8 {voice}")
        letters = list(LETTERS)
        rnd.shuffle(letters)
        names = dict(zip(("plus", "reduce", "check"), letters))
        per = {k: [] for k in names}
        log = []
        for i, (phrase, w) in enumerate(rows):
            plain = w.replace("+", "")
            base = P._with_end_punct(P._prepare_text(phrase, "+", True, True, "ru"))
            # верное ударение у слова набора - как в «ударения_38.txt»
            import re as _re
            pat = _re.compile(r"(?<![\w+])" + r"\+?".join(map(_re.escape, plain)) + r"(?![\w])",
                              _re.IGNORECASE)
            marked = pat.sub(w, base, count=1)
            reduced = pat.sub(trick(w, "reduce"), base, count=1)
            true = SD.vowel_index(w)
            seed = a.seed + 100 * i
            out = {}
            mx.random.seed(seed)
            out["plus"] = eng.synth_batch([marked], voice)[0]
            mx.random.seed(seed)
            out["reduce"] = eng.synth_batch([reduced], voice)[0]
            judged = []
            pick = None
            for att in range(4):
                if att == 0:
                    au = out["plus"]
                else:
                    mx.random.seed(seed + att)
                    au = eng.synth_batch([marked], voice)[0]
                sg = seg_in(au, phrase, plain.lower())
                pred, m = SD.by_reference(sg, refs[plain.lower()]) if sg is not None else (-1, 0)
                judged.append(pred)
                if pred == true:
                    pick = au
                    break
            out["check"] = pick if pick is not None else out["plus"]
            sg_r = seg_in(out["reduce"], phrase, plain.lower())
            red_pred = SD.by_reference(sg_r, refs[plain.lower()])[0] if sg_r is not None else -1
            log.append(f"| {i + 1} | {w} | детектор: «+» - {'верно' if judged[0] == true else 'неверно'}; "
                       f"«а/и» - {'верно' if red_pred == true else 'неверно'}; проверка - попыток "
                       f"{len(judged)}, {'нашлась верная' if pick is not None else 'верной не нашлось, взят «+»'} |")
            for k in out:
                per[k] += [out[k], np.zeros(int(1.5 * 24000), np.float32)]
            print(f"{voice}: {i + 1}/{len(rows)}", flush=True)
        folder = f"Сравнение 3.8 - ударения - {voice}"
        _os.makedirs(folder, exist_ok=True)
        for k, L in names.items():
            wav = _os.path.join(a.out_dir, f"{voice}__{k}.wav")
            sf.write(wav, np.concatenate(per[k]), 24000)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "192k",
                            _os.path.join(folder, f"Вариант {L}.mp3")], check=True)
        head = ("| № | фраза | слово (как надо) | " +
                " | ".join(f"Вариант {L}: ударение верно?" for L in LETTERS) + " |")
        table = [head, "|" + "---|" * (3 + len(LETTERS))]
        from core.stress import to_combining
        for i, (phrase, w) in enumerate(rows):
            table.append(f"| {i + 1} | {phrase} | {to_combining(w.lower())} |" + " |" * len(LETTERS))
        open(_os.path.join(folder, "Оценка.md"), "w", encoding="utf-8").write(
            f"# Ударения 3.8: {voice}\n\nВ каждом файле - одни и те же 10 фраз (пауза 1,5 с), "
            "три варианта под буквами. Слушать только ударение в слове из третьей колонки "
            "(да / нет).\n\nКлюч - в docs, открывать после оценки.\n\n" + "\n".join(table) + "\n")
        what = {"plus": "«+» перед ударной гласной (как сейчас)",
                "reduce": "«+» и безударные о → а, е → и в этом слове («+отмиль»)",
                "check": "«+» и проверка ударения по звуку: до 4 попыток с разным зерном, "
                         "берётся первая, где детектор слышит верное ударение"}
        key = [f"# Ключ 3.8 - ударения, {voice}", ""] + [f"- **Вариант {names[k]}** - {what[k]}"
                                                        for k in ("plus", "reduce", "check")]
        key += ["", "Что слышал детектор (core/stressdetect):", "", "| № | слово | итог |",
                "|---|---|---|"] + log
        open(f"docs/ключ 3.8 - ударения, {voice} (не открывать до оценки).md", "w",
             encoding="utf-8").write("\n".join(key) + "\n")
    return 0


def summary(rows):
    import collections
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["trick"], r["src"])
        by[k][0] += r["pred"] == r["true"]
        by[k][1] += 1
    srcs = sorted({r["src"] for r in rows})
    print("приём        " + "  ".join(f"{s:>10s}" for s in srcs))
    for t in dict.fromkeys(r["trick"] for r in rows):
        print(f"{t:12s} " + "  ".join(f"{by[(t, s)][0]:>4d}/{by[(t, s)][1]:<5d}" for s in srcs))
    bad = sorted({(r["word"], r["voice"]) for r in rows if r["trick"] == "plus" and r["pred"] != r["true"]})
    print("«+» не послушался:", ", ".join(f"{w} ({v.split()[0]})" for w, v in bad))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["silero", "report", "f5", "summary", "blind"])
    ap.add_argument("--out-dir", default="Архив/упрямые слова 3.8")
    ap.add_argument("--tricks", default=MAIN_TRICKS)
    ap.add_argument("--limit-rare", type=int, default=20)
    ap.add_argument("--only", default=None, help="Андрей,редкое,обычное")
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--voices", default="aidar,xenia")
    ap.add_argument("--json", default="/tmp/stubborn_silero.json")
    ap.add_argument("--save", default=None, help="папка: звук 16 кГц и кадры выравнивателя")
    a = ap.parse_args()
    if a.stage == "silero":
        return silero_stage(a)
    if a.stage == "f5":
        a.voices = a.voices if a.voices != "aidar,xenia" else "Вячеслав Герасимов,Игорь Ященко"
        return f5_stage(a)
    if a.stage == "blind":
        a.voices = a.voices if a.voices != "aidar,xenia" else "Вячеслав Герасимов,Игорь Ященко"
        _os.makedirs(a.out_dir, exist_ok=True)
        return blind_stage(a)
    if a.stage == "summary":
        summary(json.load(open(_os.path.join(a.out_dir, "итоги.json"), encoding="utf-8")))
        return 0
    report(json.load(open(a.json, encoding="utf-8")))
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
