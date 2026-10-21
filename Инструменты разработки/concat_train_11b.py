#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 3: фразы для обучения длиной как при озвучке - 15-25 с.

Учили на фразах 6,5-8,6 с, а озвучиваем кусками 35-45 с; модель могла не научиться держать
темп и голос на длинном куске. Здесь соседние фразы набора после чистки (train_clean.jsonl,
11а Д: текст сверен со звуком, в форме озвучки - со знаками ударения U+0301) склеиваются по
порядку в аудиокниге (номер файла wavs/NNNNN.wav подряд) в куски 15-25 с; между фразами -
0,35 с тишины (пауза между предложениями в конвейере). Соседство разорвано (фраза выкинута
чисткой) - кусок заканчивается.
Пишет <чтец>/concat/NNNNN.wav и <чтец>/train_concat.jsonl.

    python "Инструменты разработки/concat_train_11b.py" "Игорь Ященко"
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

LO, HI, GAP = 15.0, 25.0, 0.35


def main() -> int:
    voice = sys.argv[1] if len(sys.argv) > 1 else "Игорь Ященко"
    d = os.path.expanduser(f"~/.cache/text2audio/train/{voice}")
    rows = [json.loads(l) for l in open(f"{d}/train_clean.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["audio"])
    out_dir = os.path.join(d, "concat")
    os.makedirs(out_dir, exist_ok=True)
    groups, cur = [], []

    def num(r):
        return int(os.path.basename(r["audio"])[:-4])

    for r in rows:
        if cur and (num(r) != num(cur[-1]) + 1 or sum(x["dur"] for x in cur) + GAP + r["dur"] > HI):
            if sum(x["dur"] for x in cur) >= LO:
                groups.append(cur)
            cur = []
        cur.append(r)
    if cur and sum(x["dur"] for x in cur) >= LO:
        groups.append(cur)
    items = []
    for gi, g in enumerate(groups):
        parts, sr = [], None
        for r in g:
            a, sr = sf.read(r["audio"], dtype="float32")
            parts += [a, np.zeros(int(GAP * sr), np.float32)]
        a = np.concatenate(parts[:-1])
        wav = os.path.join(out_dir, f"{gi + 1:05d}.wav")
        sf.write(wav, a, sr)
        items.append({"audio": wav, "text": " ".join(r["text"] for r in g),
                      "text_plain": " ".join(r["text_plain"] for r in g), "ref_audio": g[0]["ref_audio"],
                      "dur": round(len(a) / sr, 2), "phrases": len(g)})
    with open(f"{d}/train_concat.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    used = sum(it["phrases"] for it in items)
    print(f"{voice}: кусков {len(items)}, {sum(it['dur'] for it in items) / 3600:.2f} ч, "
          f"фраз вошло {used} из {len(rows)}, средний кусок {np.mean([it['dur'] for it in items]):.1f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
