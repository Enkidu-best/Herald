"""Мягкое шумоподавление готового звука главы - DeepFilterNet 3 (11б, шаг 4).

У дообученного F5 Андрей слышит лёгкое шипение на заднем плане (21.10, точка В2); у
Герасимова ему понравилась чистка DeepFilterNet после синтеза (точка Б). Включается
настройкой голоса: поле «denoise» в «<голос>.fx.json» - предел подавления в дБ (0 - нет,
100 - полная сила, 12-20 - мягко: шум тише, голос не трогается).

Программа deep-filter (Rikorose/DeepFilterNet, MIT / Apache-2.0) - готовая сборка для Mac
(arm64), путь - HERALD_DFN или «.venv-restore/bin-dfn/» в папке Herald. Нет программы - звук
остаётся как есть (одно сообщение в журнал).
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile

import numpy as np

from .log import log_problem

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_warned = False


def dfn_binary() -> str | None:
    p = os.environ.get("HERALD_DFN")
    if p and os.path.isfile(p):
        return p
    found = sorted(glob.glob(os.path.join(HERE, ".venv-restore", "bin-dfn", "deep-filter-*")))
    return found[-1] if found else None


def denoise_wav(path: str, atten_db: float) -> bool:
    """WAV на месте: DeepFilterNet с пределом подавления atten_db. True - сделано."""
    global _warned
    if not atten_db:
        return False
    exe = dfn_binary()
    if not exe:
        if not _warned:
            _warned = True
            log_problem("шумоподавление голоса не сделано: нет программы deep-filter "
                        "(DeepFilterNet) - звук без него", None, show=False)
        return False
    import librosa
    import soundfile as sf
    try:
        a, sr = sf.read(path, dtype="float32")
        mono = a.mean(axis=1) if a.ndim > 1 else a
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.wav")
            sf.write(src, librosa.resample(mono, orig_sr=sr, target_sr=48000), 48000, subtype="PCM_16")
            subprocess.run([exe, "-D", "-a", str(float(atten_db)), "-o", os.path.join(d, "out"), src],
                           check=True, capture_output=True)
            b, sr2 = sf.read(os.path.join(d, "out", "in.wav"), dtype="float32")
        b = librosa.resample(b, orig_sr=sr2, target_sr=sr)[: len(mono)]
        if len(b) < len(mono):
            b = np.pad(b, (0, len(mono) - len(b)))
        sf.write(path, b.astype(np.float32), sr, subtype=sf.info(path).subtype)
        return True
    except Exception as e:
        log_problem("шумоподавление голоса не сработало - звук без него", e, show=False)
        return False
