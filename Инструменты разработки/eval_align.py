#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Размеченный набор 3.3: оценки слов по звуку на файлах, которые слушал Андрей.

Для каждого файла: звук режется по паузам на отрезки; какой кусок текста книги
звучит в отрезке, находит Whisper (ТОЛЬКО чтобы найти место - его слова в
оценку не идут); затем текст книги выравнивается по звуку (core/align) и каждое
слово получает оценки. Отмеченные Андреем места - в разметка_3.3/метки.json.

Выход - JSON со всеми словами (файл, кусок, слово, оценки, метка), по нему
eval_align_threshold.py подбирает порог.

    python "Инструменты разработки/eval_align.py" --out слова.json
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import difflib
import json
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000


def load16(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def regions(a, min_sil=0.45):
    """Отрезки речи между паузами не короче min_sil (по громкости кадров 10 мс)."""
    w = int(0.01 * SR)
    n = len(a) // w
    e = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1))
    loud = e >= max(1e-4, np.percentile(e, 90) * 0.03)
    out, start, quiet = [], None, 0
    for i, on in enumerate(loud):
        if on:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * 0.01 >= min_sil:
                out.append((start * w, (i - quiet + 1) * w))
                start, quiet = None, 0
    if start is not None:
        out.append((start * w, len(a)))
    return [(s0, s1) for s0, s1 in out if s1 - s0 > 0.3 * SR]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    import core.pipeline as P
    from core import align, asr
    if a.model:
        align._ALIGNER = align.Aligner(a.model)
    al = align.aligner().load()
    sets = json.load(open(os.path.join(HERE, "разметка_3.3", "набор.json"), encoding="utf-8"))
    rows = []
    for item in sets["файлы"]:
        path = item["звук"]
        ref = ""
        for t in item["тексты"]:
            p = t if os.path.isabs(t) else os.path.join(sets["корень"], t)
            src = open(p, encoding="utf-8").read() if os.path.exists(p) else t
            ref += P._prepare_text(src, "+", True, True, "ru") + "\n\n"
        chunks = P._chunk_paragraphs(ref, 280, 0.4, 0.8, 1.1)
        ref_words, chunk_of, ref_full = [], [], []
        for ci, c in enumerate(chunks):
            for w, _n, pn in al._words(c[0]):
                ref_words.append(w.lower().replace("ё", "е"))
                chunk_of.append(ci)
                ref_full.append(w + pn)
        audio = load16(path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            sf.write(f.name, audio, SR)
        heard = asr.transcribe(f.name, "turbo", language="ru", word_timestamps=True)
        os.unlink(f.name)
        hw = [(ww, w["start"], w["end"]) for s in heard["segments"] for w in s.get("words", [])
              for ww in P._words(w["word"])]
        sm = difflib.SequenceMatcher(None, ref_words, [x for x, _s, _e in hw], autojunk=False)
        h2r = {}
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                h2r[blk.b + k] = blk.a + k
        # прежнее правило 3.2 (текст Whisper против текста книги) - для сравнения
        old_bad = set()
        hws = [x for x, _s, _e in hw]
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag == "equal":
                continue
            for k in range(a1 - a0):
                wd = ref_words[a0 + k]
                if len(wd) < 4 or not P._known_word(ref_full[a0 + k].rstrip(",.!?…:;—–").lower()):
                    continue
                hh = hws[b0 + k] if tag == "replace" and b0 + k < b1 else ""
                if not (hh and P._same_word(wd, hh)):
                    old_bad.add(a0 + k)
        last = -1
        for r0, r1 in regions(audio):
            inside = [h2r[i] for i, (_w, s, e) in enumerate(hw)
                      if i in h2r and s * SR >= r0 - 0.15 * SR and e * SR <= r1 + 0.15 * SR]
            if not inside:
                continue
            i0, i1 = last + 1, max(inside)
            if i1 < i0:
                continue
            last = i1
            # текст отрезка - слова книги с i0 по i1 (как в тексте, с пунктуацией)
            text = " ".join(ref_full[i0:i1 + 1])
            for k, ws in enumerate(al.score(audio[r0:r1], SR, text)):
                idx = i0 + k
                rows.append({"файл": item["имя"], "кусок": f"{item['имя']}#{chunk_of[min(idx, len(chunk_of)-1)]}",
                             "i": idx, "слово": ws.word, "gop": round(ws.gop, 4),
                             "low": round(ws.low, 4), "min": round(ws.min_char, 4),
                             "gap": round(ws.gap_after, 3), "знак": ws.punct_after,
                             "длит": round(ws.end - ws.start, 3), "букв": len(ws.norm),
                             "t": round(r0 / SR + ws.start, 2),
                             "post": round(ws.post_mean, 4), "pmin": round(ws.post_min, 4),
                             "sim": round(ws.sim, 3), "слышно": ws.heard,
                             "старая": idx in old_bad, "whisper": " ".join(
                                 hws[j] for j, ri in h2r.items() if ri == idx)})
        print(item["имя"], "слов:", sum(1 for r in rows if r["файл"] == item["имя"]), flush=True)
    apply_labels(rows, sets)
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("слов всего:", len(rows), "с метками:", sum(1 for r in rows if r.get("метки")))
    return 0


def apply_labels(rows, sets):
    """Метки Андрея из набора - на слова (по контексту в тексте файла)."""
    for r in rows:
        r.pop("метки", None)
    for lab in sets["метки"]:
        ctx = [w.lower().replace("ё", "е") for w in lab["контекст"].split()]
        for fname in lab["файлы"]:
            fr = [r for r in rows if r["файл"] == fname]
            words = [r["слово"].lower().replace("ё", "е") for r in fr]
            hit = next((j for j in range(len(words) - len(ctx) + 1)
                        if words[j:j + len(ctx)] == ctx), None)
            if hit is None:
                print(f"  ! метка «{lab['контекст']}» не найдена в {fname}")
                continue
            fr[hit + lab["слово"]].setdefault("метки", []).append(
                {"id": lab["id"], "вид": lab["вид"]})
    return rows


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
