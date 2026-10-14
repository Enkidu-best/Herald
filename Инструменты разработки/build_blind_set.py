#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Собрать слепой набор для прослушивания: «Сравнение голосов 3.0/».

Все варианты проходят ОДИНАКОВУЮ финальную обработку (наш тракт HD_FILTER),
выравниваются по громкости (EBU R128, −18 LUFS) и кодируются в mp3 192 кбит/с.
Имена — случайные буквы, ключ «буква -> модель» пишется отдельно в docs.

    python "Инструменты разработки/build_blind_set.py" --runs ПАПКА_С_ПРОГОНАМИ \\
        --codes A B32 C32 D E F --out "Сравнение голосов 3.0"
"""
from __future__ import annotations

import argparse, json, os, random, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LUFS = -18.0
LETTERS = list("КЛМНПРТУФХЦШЯЖЗ")
TEXTS = [("1_повествование", "1 Повествование (книга)"),
         ("2_диалог", "2 Диалог с эмоцией"),
         ("3_сложный", "3 Сложный текст (числа, имена, омографы)")]


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def concat_phrases(run_dir: str, out_wav: str, n: int = 30, gap: float = 1.2):
    import numpy as np
    import soundfile as sf
    parts, sr = [], None
    for i in range(1, n + 1):
        a, s = sf.read(os.path.join(run_dir, f"фраза_{i:02d}.wav"), dtype="float32")
        sr = sr or s
        if s != sr:
            raise RuntimeError("разная частота у фраз")
        parts += [a.reshape(-1), np.zeros(int(gap * sr), dtype="float32")]
    sf.write(out_wav, np.concatenate(parts), sr)


def finish(src_wav: str, dst_mp3: str):
    from core.assemble import HD_FILTER
    tmp = dst_mp3 + ".tmp.wav"
    ff("-i", src_wav, "-af", HD_FILTER, "-ar", "48000", tmp)
    # двухпроходная нормализация громкости: первый проход меряет
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", tmp, "-af",
                        f"loudnorm=I={LUFS}:TP=-1.5:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = json.loads(p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1])
    ln = (f"loudnorm=I={LUFS}:TP=-1.5:LRA=11:measured_I={m['input_i']}:"
          f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
          f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    ff("-i", tmp, "-af", ln, "-ar", "48000", "-ac", "1",
       "-codec:a", "libmp3lame", "-b:a", "192k", "-write_xing", "1", dst_mp3)
    os.remove(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--codes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    letters = rnd.sample(LETTERS, len(a.codes))
    key = dict(zip(letters, a.codes))
    os.makedirs(a.out, exist_ok=True)
    for tid, folder in TEXTS + [("фразы", "4 Тридцать фраз (ударения)")]:
        os.makedirs(os.path.join(a.out, folder), exist_ok=True)
        for L, code in key.items():
            rd = os.path.join(a.runs, code)
            src = os.path.join(rd, f"{tid}.wav")
            if tid == "фразы":
                src = os.path.join(rd, "_фразы_подряд.wav")
                concat_phrases(rd, src)
            finish(src, os.path.join(a.out, folder, f"Вариант {L}.mp3"))
        print("готово:", folder, flush=True)
    json.dump(key, open(a.key, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("буквы:", " ".join(sorted(key)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
