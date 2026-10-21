#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 4: дообученный F5 - голосом в программе.

Кладёт в папку голосов F5 (~/.cache/text2audio/voices/) голос «<чтец> (F5, обучен)<хвост>»:
образец и расшифровка - те же, что у обычного голоса чтеца; «.f5model» - папка весов
(f5_finetune_11b.py: model_v1.safetensors + vocab.txt; движки f5mlx / f5torch берут их сами,
core/engines/f5_mlx.py: voice_weights); «.fx.json» - шумоподавление готового звука (denoise,
предел в дБ), если задано. Голоса чтецов - только для личного использования.

    python "Инструменты разработки/install_f5_trained_11b.py" "Игорь Ященко" <папка весов> [--denoise 15] [--tag " DFN15"]
    python "Инструменты разработки/install_f5_trained_11b.py" --remove "Игорь Ященко (F5, обучен) DFN15"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import shutil


def main() -> int:
    from core.engines.f5 import default_voices_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("reader", nargs="?")
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--denoise", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--remove", default=None)
    a = ap.parse_args()
    d = default_voices_dir()
    if a.remove:
        for ext in (".wav", ".txt", ".f5model", ".fx.json"):
            p = _os.path.join(d, a.remove + ext)
            if _os.path.exists(p):
                _os.remove(p)
        print("убран:", a.remove)
        return 0
    name = f"{a.reader} (F5, обучен){a.tag}"
    for ext in (".wav", ".txt"):
        shutil.copy(_os.path.join(d, a.reader + ext), _os.path.join(d, name + ext))
    open(_os.path.join(d, name + ".f5model"), "w", encoding="utf-8").write(_os.path.abspath(a.weights) + "\n")
    fx = _os.path.join(d, name + ".fx.json")
    if a.denoise:
        json.dump({"speed": 1.0, "pitch": 0.0, "bass": 0.0, "brightness": 0.0, "denoise": a.denoise},
                  open(fx, "w", encoding="utf-8"), indent=2)
    elif _os.path.exists(fx):
        _os.remove(fx)
    print(f"{name} -> {a.weights}" + (f", шумоподавление {a.denoise:g} дБ" if a.denoise else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
