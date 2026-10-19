#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, п. 3: интонация («пресная, на одной ноте») - сравнение на одном абзаце.

Абзац - третий длинный из набора 3.8 («Шторм кончился перед рассветом…», 641 знак).
Мерка живости - размах высоты тона: между 10-м и 90-м процентилями основного тона,
в полутонах (pyin), и его разброс; рядом - WER и ударения meter2, чтобы живость
не шла ценой ошибок. Под ../.heavy-lock.
Варианты (голос 01 папки 3.9 - образец «Голос 06» банка):
  a  как сейчас: куски до 300 знаков, temperature 0,9
  b  абзац целиком одним вызовом
  c  temperature 1,0, top_p 0,95
  d  temperature 0,8
Для наших чтецов (средний вектор): обычный образец против «живого» - фрагмента
их аудиокниги с самым большим размахом тона (Qwen перенимает манеру образца).
    python "Инструменты разработки/intonation_39.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import json

import numpy as np

BANK = "Архив/голоса диктора 3.9"
OUT = f"{BANK}/интонация"
V01 = _os.path.abspath(f"{BANK}/образцы/Голос 06.wav")


def f0_range(a: np.ndarray, sr: int) -> tuple[float, float]:
    import librosa
    y = librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a
    f0, vf, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=1024, hop_length=160)
    st = 12 * np.log2(f0[vf & np.isfinite(f0)] / 100.0)
    if len(st) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(st, 90) - np.percentile(st, 10)), float(np.std(st))


def synth(eng, text: str, voice: str, chunk: int) -> np.ndarray:
    import core.pipeline as P
    pieces = P._chunk_paragraphs(text, chunk, 0.4, 0.8, 1.1) if len(text) > chunk else [(text, 0.0, 1.0, ())]
    parts = []
    for pc in pieces:
        parts += [eng.synth_chunk(P._with_end_punct(pc[0]), voice),
                  np.zeros(int(pc[1] * eng.sample_rate), np.float32)]
    return np.concatenate(parts)


def lively_fragment(voice: str) -> str:
    """Фрагмент аудиокниги чтеца с самым большим размахом тона + его расшифровка."""
    import soundfile as sf
    import core.pipeline as P
    best = None
    for f in sorted(glob.glob(f"{BANK}/фрагменты/{voice}/*.wav")):
        a, sr = sf.read(f, dtype="float32")
        r, _s = f0_range(a, sr)
        if best is None or r > best[0]:
            best = (r, f, a, sr)
    r, f, a, sr = best
    dst = _os.path.abspath(f"{OUT}/живой образец - {voice}.wav")
    _os.makedirs(OUT, exist_ok=True)
    sf.write(dst, a, sr)
    h = P._listen(a, sr, "ru", P._ASR_FULL)
    open(dst[:-4] + ".txt", "w", encoding="utf-8").write(" ".join(w for w, _s, _e in (h or [])))
    return dst


def main() -> int:
    import soundfile as sf
    import core.pipeline as P
    from core import meter
    from core.engines.mlxaudio import MlxAudioEngine
    import importlib.util as eng_mod
    spec = eng_mod.spec_from_file_location("e", "Инструменты разработки/bench_engines_38.py")
    e = eng_mod.module_from_spec(spec)
    spec.loader.exec_module(e)
    rows = e.rows()
    para = rows[19][0]
    text = P._with_end_punct(P._prepare_text(para, "+", True, True, "ru"))
    _os.makedirs(OUT, exist_ok=True)
    runs = [("a: как сейчас", V01, {}, 300), ("b: абзац целиком", V01, {}, 2000),
            ("c: temperature 1,0, top_p 0,95", V01, {"temperature": 1.0, "top_p": 0.95}, 300),
            ("d: temperature 0,8", V01, {"temperature": 0.8}, 300)]
    for v in ("Вячеслав Герасимов", "Игорь Ященко"):
        frag = _os.path.abspath(f"{BANK}/фрагменты/{v}")
        runs.append((f"{v}: обычный образец", v, {"centroid": frag}, 300))
        runs.append((f"{v}: живой образец", lively_fragment(v), {"centroid": frag}, 300))
    res = {}
    print("| вариант | размах тона, полутонов | разброс тона | WER | ударения |")
    print("|---|---|---|---|---|")
    for name, voice, kw, chunk in runs:
        eng = MlxAudioEngine(model="qwen-stress", bits=8, seed=7, **kw)
        eng.ensure_loaded()
        au = synth(eng, text, voice, chunk)
        eng.close()
        sf.write(f"{OUT}/{name.replace(':', ' -')}.wav", au, eng.sample_rate)
        rng, sd = f0_range(au, eng.sample_rate)
        wer = e.wer_of(au, eng.sample_rate, para)[0]
        bad = meter.wrong_words(au, eng.sample_rate, text)
        res[name] = {"размах": round(rng, 2), "разброс": round(sd, 2), "wer": round(wer, 3), "ударения не так": bad}
        print(f"| {name} | {rng:.1f} | {sd:.2f} | {wer:.1%} | {len(bad)} не так |", flush=True)
    json.dump(res, open(f"{OUT}/итог.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
