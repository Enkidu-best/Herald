#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение вариантов скорости на одном и том же тексте — чтобы выбрать на слух.

Складывает mp3 в папку «Сравнение скорости» и печатает RTF каждого варианта.
Голос, образец и ударения во всех вариантах одни и те же; меняется только
расписание шагов. epss-7 — прорежённое расписание из Fast F5-TTS (arXiv
2505.19931): 7 шагов вместо 15 при том же качестве. uniform-16 — эталон,
в точности как torch-версия.

Модель поднимается ОДИН раз, а число шагов меняется между прогонами: иначе на
каждый вариант грузилась бы своя копия весов (1,3 ГБ) и на 16 ГБ памяти
становилось тесно, а MLX падал при выходе.

    python compare_speed.py
    python compare_speed.py --chapter 2 --modes epss7,uniform16
"""
from __future__ import annotations

import os as _os, sys as _sys
# скрипт лежит в подпапке, а core/ — в корне проекта
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import os, time, argparse, subprocess

_hs = os.environ.get("PYTHONHASHSEED")
if _hs is None or not _hs.isdigit():
    os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import soundfile as sf

from core.engines import get_engine
from core.readers import read_document
from core.chapters import build_chapters
from core.normalize import normalize_text, sentenize
from core.stress import add_stress
from core.pipeline import _chunk_sentences, _synth_chunks, CHARS_PER_SEC

OUT_DIR = "Сравнение скорости"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book", nargs="?", default="Пример — проба пера.txt")
    ap.add_argument("--voice", default="голос диктора")
    ap.add_argument("--chapter", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=0.5)
    ap.add_argument("--modes", default="epss7,epss6,epss10,uniform16",
                    help="что сравнивать: epss7 | epss6 | epss10 | epss12 | uniform16")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    def parse(name):
        """epss7 -> ("epss", 7); uniform16 -> ("uniform", 16)"""
        sch = "uniform" if name.startswith("uniform") else "epss"
        return sch, int(name.replace("uniform", "").replace("epss", ""))
    modes = [parse(x.strip()) for x in args.modes.split(",")]

    # один и тот же текст для всех вариантов
    doc = read_document(args.book)
    chapters = build_chapters(doc)
    ch = chapters[max(0, min(args.chapter - 1, len(chapters) - 1))]
    eng = get_engine("f5mlx", steps=modes[0][1], schedule=modes[0][0])
    text = add_stress(normalize_text(ch.text, expand_numbers=True))
    sents = sentenize(text)
    budget = int(args.minutes * 60 * CHARS_PER_SEC)
    picked, total, i = [], 0, len(sents) // 2
    while i < len(sents) and total < budget:
        picked.append(sents[i]); total += len(sents[i]); i += 1
    chunks = _chunk_sentences(" ".join(picked or sents[:1]), eng.max_chunk_chars)

    print(f"глава «{ch.title or args.chapter}», кусков: {len(chunks)}", flush=True)
    eng.ensure_loaded()
    rows = []
    for sch, nfe in modes:
        eng.schedule, eng.steps = sch, nfe    # модель уже в памяти, меняем только шаги
        t0 = time.time()
        audio = _synth_chunks(eng, args.voice, chunks, eng.sample_rate, 0.35)
        elapsed = time.time() - t0
        dur = len(audio) / eng.sample_rate
        rtf = elapsed / dur if dur else float("inf")
        wav = os.path.join(OUT_DIR, f"_tmp_{sch}{nfe}.wav")
        sf.write(wav, audio, eng.sample_rate)
        mp3 = os.path.join(OUT_DIR, f"{sch}-{nfe} — RTF {rtf:.2f}.mp3")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav,
                        "-codec:a", "libmp3lame", "-b:a", "128k", "-write_xing", "1", mp3],
                       check=True)
        os.remove(wav)
        rows.append((f"{sch}-{nfe}", rtf))
        print(f"  {sch}-{nfe}: звук {dur:5.1f} c | счёт {elapsed:5.1f} c | RTF {rtf:.2f}", flush=True)

    print(f"\nФайлы в папке «{OUT_DIR}» — послушай и скажи, какой вариант годится.")
    print("Сколько займёт час звука:")
    for name, rtf in rows:
        print(f"  {name:12s}: {rtf*60:4.0f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
