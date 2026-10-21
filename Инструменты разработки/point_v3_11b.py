#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б (решение Андрея 21.10 16:05), точка В3: проба F5 Герасимова, обученного на «Олесе».

  1) образец «Олеси» - чистая фраза 7-10 с, ближе всех (ECAPA) к среднему голосу книги ->
     голос «Вячеслав Герасимов (Олеся)» в папке голосов F5;
  2) два пробных голоса с одними весами пробы: образец «Олеся» и образец «К» (принятый);
  3) тот же абзац полным конвейером: проба + Олеся, проба + К; к ним - «F5 как сейчас (К)» из
     точки А и живой Герасимов (минута «Олеси»);
  4) замер (measure_11b.py) и шум в паузах; лучшей пробе по похожести - ещё вариант с чисткой
     DeepFilterNet после синтеза (Андрею у Герасимова она понравилась, точка Б).
Пишет «Сравнение 3.10 - лучшие варианты/Точка В3/». Пробные голоса после - убираются. Под ../.heavy-lock.

    python "Инструменты разработки/point_v3_11b.py" <папка весов пробы>
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import importlib.util
import json
import shutil
import subprocess

import numpy as np

R = "Вячеслав Герасимов"
DST = "Сравнение 3.10 - лучшие варианты/Точка В3"
OLESYA = "Образцы голосов/Вячеслав Герасимов 3. Куприн - Олеся.m4b"
T = "Инструменты разработки"
PY = _sys.executable


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(*args) -> str:
    r = subprocess.run([PY, *args], capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout or "").splitlines() if not l.lower().startswith("warn"))
    print(out, flush=True)
    return out


def pick_ref() -> str:
    from core.engines.f5 import default_voices_dir
    sim = _mod("sim38", f"{T}/bench_similarity_38.py")
    sim.KIND = "ecapa"
    d = _os.path.expanduser(f"~/.cache/text2audio/train/{R}")
    rows = [json.loads(l) for l in open(f"{d}/train_olesya_clean.jsonl", encoding="utf-8")]
    cand = [r for r in rows if 7.0 <= r["dur"] <= 10.0][:200]
    E = np.array([sim.emb(sim.load16(r["audio"])) for r in cand])
    c = E.mean(axis=0)
    c /= np.linalg.norm(c)
    best = cand[int(np.argmax(E @ c))]
    vd = default_voices_dir()
    shutil.copy(best["audio"], _os.path.join(vd, f"{R} (Олеся).wav"))
    open(_os.path.join(vd, f"{R} (Олеся).txt"), "w", encoding="utf-8").write(best["text_plain"].strip() + "\n")
    print("образец «Олеси»:", best["audio"], best["dur"], "с:", best["text_plain"], flush=True)
    return best["audio"]


def main() -> int:
    weights = _sys.argv[1]
    _os.makedirs(DST, exist_ok=True)
    pick_ref()
    probes = {"Олеся": f"{R} (F5, проба, образец Олеся)", "К": f"{R} (F5, проба, образец К)"}
    run(f"{T}/install_f5_trained_11b.py", R, weights, "--ref", f"{R} (Олеся)", "--name", probes["Олеся"])
    run(f"{T}/install_f5_trained_11b.py", R, weights, "--name", probes["К"])
    for tag, voice in probes.items():
        run(f"{T}/paragraph_11b.py", "Точка В3", "main", voice, f"Герасимов - проба Олеся + образец {tag}")
    shutil.copy("Сравнение 3.10 - лучшие варианты/Точка А/Герасимов - F5.mp3",
                _os.path.join(DST, "Герасимов - F5 как сейчас (образец К).mp3"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1800", "-t", "60", "-i", OLESYA, "-ac", "1",
                    "-b:a", "192k", _os.path.join(DST, "Герасимов - живой (Олеся, 30-я минута).mp3")], check=True)
    files = [("Герасимов - F5 как сейчас (образец К).mp3", "F5 как сейчас (К)"),
             ("Герасимов - проба Олеся + образец Олеся.mp3", "проба + образец Олеся"),
             ("Герасимов - проба Олеся + образец К.mp3", "проба + образец К")]
    out = run(f"{T}/measure_11b.py", "точка В3", R, "Архив/11б/абзац.txt",
              *[f"{DST}/{f}={lab}" for f, lab in files])
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    probe_rows = [r for r in rows if r["вариант"].startswith("проба")]
    best = max(probe_rows, key=lambda r: r["похожесть"] or 0)
    tag = "Олеся" if "Олеся" in best["вариант"] else "К"
    run(f"{T}/install_f5_trained_11b.py", R, weights, "--ref", f"{R} (Олеся)" if tag == "Олеся" else R,
        "--name", probes[tag] + " DFN", "--denoise", "100")
    run(f"{T}/paragraph_11b.py", "Точка В3", "main", probes[tag] + " DFN",
        f"Герасимов - проба Олеся + образец {tag} + чистка DFN")
    run(f"{T}/measure_11b.py", "точка В3", R, "Архив/11б/абзац.txt",
        f"{DST}/Герасимов - проба Олеся + образец {tag} + чистка DFN.mp3=проба + образец {tag} + DFN")
    for v in list(probes.values()) + [probes[tag] + " DFN"]:
        run(f"{T}/install_f5_trained_11b.py", "--remove", v)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
