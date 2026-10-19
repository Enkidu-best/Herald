#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, шаг 1: ровный голос у Qwen ru-stress (Herald 3.8 -> 3.9).

Этапы (тяжёлые - под ../.heavy-lock):
  pieces    - «голоса диктора» самой ru-stress: длинные абзацы записей ru-stress
              (первый набор 3.8, записи 1, 2, 5, 9 = qwen / qwen4, оба чтеца) режутся
              по вставленным паузам на куски одного вызова (один вызов - один голос),
              у каждого - вектор голоса ECAPA, шумность (спектральная плоскостность
              6-11 кГц на речи) и расшифровка Whisper. Куски группируются по голосу,
              из каждой группы - самый чистый, 12-15 с -> образец «Голос NN».
  centroid  - вектор голоса чтеца, усреднённый по 60 чистым фрагментам его
              аудиокниг (Образцы голосов/): для режима «только вектор голоса».
Синтез и замеры - voice_synth_39.py (окружение mlx-audio) и measure_voice_stress.py.

    python "Инструменты разработки/voice_bank_39.py" pieces | centroid
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import importlib.util
import json

import numpy as np

SET = "Архив/сравнение движков 3.8"
BANK = "Архив/голоса диктора 3.9"
VOICES = ("Вячеслав Герасимов", "Игорь Ященко")


def _sim():
    spec = importlib.util.spec_from_file_location("sim", "Инструменты разработки/bench_similarity_38.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.KIND = "ecapa"
    return m


def split_on_zeros(a: np.ndarray, sr: int, min_gap: float = 0.3) -> list[np.ndarray]:
    """Куски между вставленными паузами (ровно нули, >= min_gap с)."""
    z = np.concatenate([[False], np.abs(a) < 1e-7, [False]])
    edges = np.flatnonzero(np.diff(z.astype(np.int8)))
    runs = [(s0, e0) for s0, e0 in zip(edges[::2], edges[1::2]) if e0 - s0 >= int(min_gap * sr)]
    out, pos = [], 0
    for s0, e0 in runs:
        if s0 - pos > sr:
            out.append(a[pos:s0])
        pos = e0
    if len(a) - pos > sr:
        out.append(a[pos:])
    return out


def hiss(a: np.ndarray, sr: int) -> float:
    """Шумность: средняя спектральная плоскостность 6-11 кГц на громких кадрах."""
    from scipy.signal import stft
    f, _t, Z = stft(a, sr, nperseg=1024, noverlap=768)
    P = np.abs(Z) ** 2
    band = (f >= 6000) & (f <= 11000)
    loud = P.sum(0) > np.percentile(P.sum(0), 60)
    B = P[band][:, loud] + 1e-12
    flat = np.exp(np.log(B).mean(0)) / B.mean(0)
    return float(np.median(flat))


def pieces() -> int:
    import soundfile as sf
    import core.pipeline as P
    sim = _sim()
    rows = []
    for eng in ("qwen", "qwen4"):
        for v in VOICES:
            for i in (18, 19, 20):                       # длинные абзацы
                a, sr = sf.read(_os.path.join(SET, eng, f"{v}__{i:02d}.wav"), dtype="float32")
                for k, seg in enumerate(split_on_zeros(a, sr)):
                    d = len(seg) / sr
                    if d < 6:
                        continue
                    from math import gcd
                    from scipy.signal import resample_poly
                    g = gcd(sr, 16000)
                    x16 = resample_poly(seg, 16000 // g, sr // g).astype(np.float32)
                    h = P._listen(seg, sr, "ru", P._ASR_FULL)
                    text = " ".join(w for w, _s, _e in (h or []))
                    rows.append({"src": f"{eng}|{v}|{i}|{k}", "dur": round(d, 1),
                                 "hiss": round(hiss(seg, sr), 4), "text": text,
                                 "emb": sim.emb(x16).tolist(), "_a": seg, "_sr": sr})
    print(f"кусков: {len(rows)}", flush=True)
    # группы по голосу: жадно, порог похожести ECAPA 0,6
    E = np.array([r["emb"] for r in rows])
    order = sorted(range(len(rows)), key=lambda j: rows[j]["hiss"])     # чище - раньше
    groups: list[list[int]] = []
    for j in order:
        for g in groups:
            if float(E[j] @ E[g[0]]) >= 0.6:
                g.append(j)
                break
        else:
            groups.append([j])
    groups.sort(key=lambda g: (-len(g), rows[g[0]]["hiss"]))
    _os.makedirs(_os.path.join(BANK, "образцы"), exist_ok=True)
    info = []
    n = 0
    for g in groups:
        best = min(g, key=lambda j: rows[j]["hiss"])
        r = rows[best]
        if not r["text"] or r["dur"] < 8:
            continue
        n += 1
        a, sr = r["_a"], r["_sr"]
        if len(a) > 15 * sr:                             # образец 12-15 с - срез в самом тихом месте
            fr = int(0.02 * sr)
            lo, hi = int(12 * sr) // fr, int(15 * sr) // fr
            rms = [float(np.sqrt(np.mean(a[k * fr:(k + 1) * fr] ** 2))) for k in range(lo, hi)]
            a = a[: (lo + int(np.argmin(rms))) * fr + fr // 2]
        name = f"Голос {n:02d}"
        sf.write(_os.path.join(BANK, "образцы", name + ".wav"), a, sr)
        h = P._listen(a, sr, "ru", P._ASR_FULL)
        txt = " ".join(w for w, _s, _e in (h or []))
        open(_os.path.join(BANK, "образцы", name + ".txt"), "w", encoding="utf-8").write(txt)
        info.append({"голос": name, "источник": r["src"], "кусков в группе": len(g),
                     "шумность": r["hiss"], "длина": round(len(a) / sr, 1), "текст": txt})
        print(f"{name}: группа {len(g)} кусков, источник {r['src']}, шумность {r['hiss']}, "
              f"{len(a) / sr:.1f} с", flush=True)
        if n >= 6:
            break
    json.dump(info, open(_os.path.join(BANK, "образцы.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


def centroid() -> int:
    """Вектор голоса чтеца по 60 фрагментам его аудиокниг: кусками 8 с подряд,
    с отбором по громкости (без тишины) - для mlx-audio (вектор голоса Qwen
    считается его собственным кодировщиком из этих же кусков - voice_synth_39)."""
    import subprocess
    srcs = {"Вячеслав Герасимов": ["Образцы голосов/Вячеслав Герасимов 2.mp3",
                                   "Образцы голосов/Вячеслав Герасимов.mp3"],
            "Игорь Ященко": sorted(glob.glob("Образцы голосов/Каменистый Артем - Альфа-ноль 11, "
                                             "Убийца легенд/00*.mp3"))[:6]}
    for v, files in srcs.items():
        d = _os.path.join(BANK, "фрагменты", v)
        _os.makedirs(d, exist_ok=True)
        n = 0
        for f in files:
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                        "-of", "csv=p=0", f], capture_output=True, text=True).stdout)
            t = 30.0
            while t + 8 < dur and n < 60:
                out = _os.path.join(d, f"{n:02d}.wav")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-t", "8", "-i", f,
                                "-ac", "1", "-ar", "24000", out], check=True)
                import soundfile as sf
                a, sr = sf.read(out, dtype="float32")
                if np.mean(np.abs(a) > 0.02) < 0.5:          # мало речи - пропустить
                    _os.remove(out)
                else:
                    n += 1
                t += 20.0 if v == "Вячеслав Герасимов" else 60.0
            if n >= 60:
                break
        print(f"{v}: фрагментов {n}", flush=True)
    return 0


if __name__ == "__main__":
    stage = _sys.argv[1] if len(_sys.argv) > 1 else "pieces"
    rc = {"pieces": pieces, "centroid": centroid}[stage]()
    _sys.stdout.flush()
    _os._exit(rc)
