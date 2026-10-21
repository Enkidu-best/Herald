#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 2: чистота F5 у источника - восстановить ОБРАЗЕЦ чтеца до синтеза
(F5 копирует акустику образца: шум, сжатие mp3 аудиокниги).

Способы (окружение .venv-restore, только CPU):
  Sidon        - восстановление речи (sarulab-speech/Sidon, MIT, русский проверен; ONNX через
                 audiosronnx), 16 кГц -> 48 кГц;
  DFN          - обычный шумодав DeepFilterNet 3 (MIT/Apache, готовая программа deep-filter);
  Sidon70      - смесь: 70 % Sidon + 30 % исходного (против потери похожести - риск из
                 практики: WildSpoof 2026 на F5 - чище, но похожесть 0,35 -> 0,28).
Результат - 24 кГц (частота F5), голос «<чтец> (<способ>)» рядом с исходным образцом в
папке голосов F5: тот же текст образца. Голоса чтецов - только для личного использования.

    .venv-restore/bin/python "Инструменты разработки/restore_ref_11b.py" [wav ...]
    (без аргументов - образцы «Игорь Ященко» и «Вячеслав Герасимов»)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import librosa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES = os.path.expanduser("~/.cache/text2audio/voices")
DFN = os.path.join(HERE, ".venv-restore", "bin-dfn", "deep-filter-0.5.6-aarch64-apple-darwin")
READERS = ("Игорь Ященко", "Вячеслав Герасимов")
_sidon = None


def sidon(a: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    global _sidon
    from audiosronnx import load_sr
    if _sidon is None:
        _sidon = load_sr("sidon")
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        sf.write(i, a, sr)
        _sidon.upscale_file(i, o)
        b, sr2 = sf.read(o, dtype="float32")
    return (b.mean(axis=1) if b.ndim > 1 else b), sr2


def dfn(a: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as d:
        i = os.path.join(d, "in.wav")
        sf.write(i, librosa.resample(a, orig_sr=sr, target_sr=48000), 48000, subtype="PCM_16")
        subprocess.run([DFN, "-o", d + "/out", i], check=True, capture_output=True)
        b, sr2 = sf.read(os.path.join(d, "out", "in.wav"), dtype="float32")
    return b, sr2


def to24(a, sr):
    b = librosa.resample(a, orig_sr=sr, target_sr=24000) if sr != 24000 else a
    return (b / max(1.0, float(np.abs(b).max()) / 0.98)).astype(np.float32)


def main() -> int:
    srcs = sys.argv[1:] or [os.path.join(VOICES, r + ".wav") for r in READERS]
    for src in srcs:
        name = os.path.basename(src)[:-4]
        a, sr = sf.read(src, dtype="float32")
        a = a.mean(axis=1) if a.ndim > 1 else a
        orig24 = to24(a, sr)
        s, ssr = sidon(a, sr)
        s24 = to24(s, ssr)
        n = min(len(s24), len(orig24))
        out = {"Sidon": s24, "DFN": to24(*dfn(a, sr)),
               "Sidon70": to24(0.7 * s24[:n] + 0.3 * orig24[:n], 24000)}
        for tag, b in out.items():
            dst = os.path.join(VOICES, f"{name} ({tag})")
            sf.write(dst + ".wav", b, 24000)
            shutil.copy(os.path.join(VOICES, name + ".txt"), dst + ".txt")
            print(f"{name} ({tag}): {len(b) / 24000:.1f} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
