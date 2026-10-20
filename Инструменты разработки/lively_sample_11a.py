#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Б: «живой» образец для голосов 01 и 02 («на одной ноте»).

У этих голосов в банке 3.9 один кусок на группу - брать другой фрагмент негде. Зато
есть их длинный кусок (8 мин тем же голосом, полный конвейер). Из него - окно
12-15 с от паузы до паузы с самым большим размахом тона (10-90 процентили, полутоны)
среди окон, где голос тот же (ECAPA к образцу не ниже 0,75). Расшифровка - текст
оригинала с пунктуацией (Whisper находит место, текст берётся из книги): с голой
расшифровкой модель чаще выдаёт пустоту (3.9).
Сравнение на абзаце «Шторм кончился…» (как в п. 3 задания 11): обычный / живой
образец с настройками Qwen по умолчанию (куски до 600 знаков, temperature 1,0,
top_p 0,95, одно зерно) - размах тона, WER, ударения meter2.
Файлы на прослушивание - «Сравнение 3.9 - голоса диктора на выбор/Живой образец/».
Под ../.heavy-lock.
    python "Инструменты разработки/lively_sample_11a.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import difflib
import importlib.util
import json
import re

import numpy as np

BANK = "Архив/голоса диктора 3.9"
SRC = f"{BANK}/длинный кусок/Глава первая (отрывок).txt"
DST = "Сравнение 3.9 - голоса диктора на выбор/Живой образец"
OUT = f"{BANK}/живой образец"
VOICES = {"01": "Голос 06", "02": "Голос 05"}        # номер в папке Андрея -> образец банка


def _mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runs(a: np.ndarray, sr: int, min_gap: float = 0.3) -> list[tuple[int, int]]:
    """Отрезки речи между паузами не короче min_gap (тишина - ниже -40 дБ от пика)."""
    fr = int(0.01 * sr)
    n = len(a) // fr
    rms = np.sqrt(np.mean(a[: n * fr].reshape(n, fr) ** 2, axis=1)) + 1e-9
    loud = 20 * np.log10(rms / rms.max()) > -40
    segs, start, quiet = [], None, 0
    for i, v in enumerate(loud):
        if v:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * 0.01 >= min_gap:
                segs.append((start * fr, (i - quiet + 1) * fr))
                start, quiet = None, 0
    if start is not None:
        segs.append((start * fr, n * fr))
    return segs


def semitones(a16: np.ndarray) -> np.ndarray:
    import librosa
    f0, vf, _ = librosa.pyin(a16, fmin=60, fmax=400, sr=16000, frame_length=1024, hop_length=160)
    st = np.full(len(f0), np.nan)
    ok = vf & np.isfinite(f0)
    st[ok] = 12 * np.log2(f0[ok] / 100.0)
    return st                                  # по кадру на 10 мс


def norm_words(s: str) -> list[str]:
    return re.findall(r"[а-яё0-9]+", s.lower().replace("ё", "е"))


def source_text(heard: str, src: str) -> str:
    """Место в оригинале, которое слышно в окне, - с пунктуацией оригинала."""
    toks = [(m.group().lower().replace("ё", "е"), m.start(), m.end())
            for m in re.finditer(r"[А-Яа-яЁё0-9]+", src)]
    words = [t[0] for t in toks]
    hyp = norm_words(heard)
    best = (0.0, 0)
    for i in range(0, max(1, len(words) - len(hyp))):
        r = difflib.SequenceMatcher(None, hyp, words[i:i + len(hyp)], autojunk=False).ratio()
        if r > best[0]:
            best = (r, i)
    i, j = best[1], min(len(toks) - 1, best[1] + len(hyp) - 1)
    end = toks[j][2]
    while end < len(src) and src[end] in ".,!?…;:»\"":
        end += 1
    return src[toks[i][1]:end].strip(" -–—")


def pick_lively(num: str) -> dict:
    import soundfile as sf
    import core.pipeline as P
    sim = _mod("sim38", "Инструменты разработки/bench_similarity_38.py")
    sim.KIND = "ecapa"
    mp3 = f"Сравнение 3.9 - голоса диктора на выбор/Длинный кусок/Голос {num}.mp3"
    ref = f"{BANK}/образцы/{VOICES[num]}.wav"
    a = sim.load16(mp3)
    e_ref = sim.emb(sim.load16(ref))
    st = semitones(a)
    segs = runs(a, 16000)
    cands = []
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            dur = (segs[j][1] - segs[i][0]) / 16000
            if dur > 15.0:
                break
            if dur >= 12.0:
                s, e = segs[i][0], segs[j][1]
                f = st[s // 160: e // 160]
                f = f[np.isfinite(f)]
                if len(f) > 200:
                    cands.append((float(np.percentile(f, 90) - np.percentile(f, 10)), s, e))
    cands.sort(reverse=True)
    best = None
    for rng, s, e in cands[:40]:                  # самые живые - проверяем, тот ли голос
        c = float(np.dot(sim.emb(a[s:e]), e_ref))
        if c >= 0.75:
            best = (rng, s, e, c)
            break
    if best is None:
        rng, s, e = cands[0]
        best = (rng, s, e, float(np.dot(sim.emb(a[s:e]), e_ref)))
    rng, s, e, c = best
    # образец - в исходных 24 кГц (16 кГц - только для замеров): модель перенимает и звук образца
    import subprocess
    a24 = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-ac", "1", "-ar", "24000",
                                        "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)
    clip = np.concatenate([np.zeros(2400, np.float32), a24[s * 3 // 2:e * 3 // 2], np.zeros(4800, np.float32)])
    heard = " ".join(w for w, _s, _e in (P._listen(clip, 24000, "ru", P._ASR_FULL) or []))
    text = source_text(heard, open(SRC, encoding="utf-8").read())
    _os.makedirs(OUT, exist_ok=True)
    path = _os.path.abspath(f"{OUT}/Голос {num} - живой.wav")
    sf.write(path, clip, 24000)
    open(path[:-4] + ".txt", "w", encoding="utf-8").write(text + "\n")
    f_ref = semitones(sim.load16(ref))
    f_ref = f_ref[np.isfinite(f_ref)]
    info = {"образец": path, "с": round((e - s) / 16000, 1), "размах тона окна": round(rng, 1),
            "размах тона обычного образца": round(float(np.percentile(f_ref, 90) - np.percentile(f_ref, 10)), 1),
            "похожесть на обычный образец (ECAPA)": round(c, 2), "текст": text, "слышно": heard}
    print(num, json.dumps(info, ensure_ascii=False), flush=True)
    return info


def compare(num: str, lively: str) -> dict:
    import soundfile as sf
    import core.pipeline as P
    from core import meter
    from core.engines.mlxaudio import MlxAudioEngine
    ito = _mod("ito39", "Инструменты разработки/intonation_39.py")
    e = _mod("e38", "Инструменты разработки/bench_engines_38.py")
    para = e.rows()[19][0]
    text = P._with_end_punct(P._prepare_text(para, "+", True, True, "ru"))
    res = {}
    _os.makedirs(DST, exist_ok=True)
    for kind, voice in (("обычный образец", _os.path.abspath(f"{BANK}/образцы/{VOICES[num]}.wav")),
                        ("живой образец", lively)):
        eng = MlxAudioEngine(model="qwen-stress", bits=8, seed=7, temperature=1.0, top_p=0.95, max_chars=600)
        eng.ensure_loaded()
        au = ito.synth(eng, text, voice, 600)
        eng.close()
        sf.write(f"{DST}/Голос {num} - {kind}.wav", au, eng.sample_rate)
        rng, sd = ito.f0_range(au, eng.sample_rate)
        wer = e.wer_of(au, eng.sample_rate, para)[0]
        bad = meter.wrong_words(au, eng.sample_rate, text)
        res[kind] = {"размах тона": round(rng, 1), "WER": round(wer, 3), "ударения не так": bad}
        print(num, kind, res[kind], flush=True)
    return res


def main() -> int:
    nums = _sys.argv[1:] or list(VOICES)
    out = {}
    for num in nums:
        info = pick_lively(num)
        info["сравнение"] = compare(num, info["образец"])
        out[num] = info
    path = f"{OUT}/итог.json"
    old = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    old.update(out)
    json.dump(old, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
