#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Д: сколько слов теряет обученный голос - варианты модели и выборки.

Начало «Главы первой» (~900 знаков, 5 кусков по 600, 302 слова) - голосом из списка
движка Qwen (обученные: «<чтец> (обучен…)»), без пересчётов конвейера; распознавание
Whisper large-v3-turbo, сверка слов с текстом. Похожесть - ECAPA к среднему вектору
настоящих фрагментов чтеца. Темп - слогов в секунду звука без пауз.
Варианты: «голос|temperature|top_p|top_k» (top_k 0 - как в профиле). Под ../.heavy-lock.

    python "Инструменты разработки/trained_words_11a.py" "Игорь Ященко (обучен)|1.0|0.95|0" ...
Пишет «Архив/голоса диктора 3.9/обученный голос/слова.jsonl» (по строке на вариант).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import difflib
import glob
import importlib.util
import json
import re
import time

import numpy as np

BANK = "Архив/голоса диктора 3.9"
OUT = f"{BANK}/обученный голос"
VOW = "аеёиоуыэюя"


def _sim():
    spec = importlib.util.spec_from_file_location("sim38", "Инструменты разработки/bench_similarity_38.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    s.KIND = "ecapa"
    return s


def opening() -> str:
    src = open(f"{BANK}/длинный кусок/Глава первая (отрывок).txt", encoding="utf-8").read()
    txt = ""
    for p in [p for p in src.split("\n\n") if p.strip()]:
        if len(txt) > 900:
            break
        txt += p + "\n\n"
    return txt


def check(P, a16, text):
    h = P._listen(a16, 16000, "ru", P._ASR_FULL) or []
    sw = [w.lower().replace("ё", "е") for w in re.findall(r"[А-Яа-яЁё]+", text.replace("+", ""))]
    hw = [re.sub(r"[^а-яё]", "", w.lower()).replace("ё", "е") for w, _a, _b in h]
    sm = difflib.SequenceMatcher(None, sw, hw, autojunk=False)
    lost = [" ".join(sw[i1:i2]) for tag, i1, i2, _j1, _j2 in sm.get_opcodes()
            if tag in ("delete", "replace") and i2 - i1 >= 2]
    return sum(b.size for b in sm.get_matching_blocks()), len(sw), lost, sw


def speech_sec(a16) -> float:
    fr = 160
    rms = np.sqrt(np.mean(a16[: len(a16) // fr * fr].reshape(-1, fr) ** 2, axis=1)) + 1e-9
    return float(np.sum(20 * np.log10(rms / rms.max()) > -35)) * 0.01


def main() -> int:
    import librosa
    import soundfile as sf
    import core.pipeline as P
    from core.engines.mlxaudio import MlxAudioEngine
    from core.profiles import PROFILES, engine_kwargs
    sim = _sim()
    chunks = P._chunk_paragraphs(P._prepare_text(opening(), "+", True, True, "ru"), 600, 0.4, 0.8, 1.1)
    _os.makedirs(f"{OUT}/слова", exist_ok=True)
    refs = {}
    for spec in _sys.argv[1:]:
        voice, temp, top, tk = spec.split("|")
        reader = voice.split(" (")[0]
        if reader not in refs:
            E = [sim.emb(sim.load16(f)) for f in sorted(glob.glob(f"{BANK}/фрагменты/{reader}/*.wav"))]
            r = np.mean(E, axis=0)
            refs[reader] = r / np.linalg.norm(r)
        kw = engine_kwargs(PROFILES["qwen"])
        kw["temperature"], kw["top_p"] = float(temp), float(top)
        eng = MlxAudioEngine(**kw)
        if int(tk):
            eng.extra["top_k"] = int(tk)
        eng.ensure_loaded()
        sr = eng.sample_rate
        M = N = syl = 0
        sp, sims, lost, parts = 0.0, [], [], []
        t0 = time.time()
        for c in chunks:
            a = eng.synth_chunk(c[0], voice)
            a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
            m, n, l, sw = check(P, a16, c[0])
            M, N, lost = M + m, N + n, lost + l
            syl += sum(sum(ch in VOW for ch in w) for w in sw)
            sp += speech_sec(a16)
            sims.append(float(sim.emb(a16) @ refs[reader]))
            parts += [a, np.zeros(int(0.5 * sr), np.float32)]
        eng.close()
        tag = f"{voice} t{temp} p{top} k{tk}"
        sf.write(f"{OUT}/слова/{tag}.wav", np.concatenate(parts), sr)
        res = {"вариант": tag, "слов": M, "из": N, "%": round(100 * M / N, 1),
               "похожесть": round(float(np.mean(sims)), 3), "слогов в секунду речи": round(syl / max(sp, 1e-6), 2),
               "пропуски": lost, "с": round(time.time() - t0)}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open(f"{OUT}/слова.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
