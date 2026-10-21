#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Д: обученные голоса чтецов - в список голосов движка Qwen ru-stress.

Для каждого чтеца, у которого есть модель ~/.cache/text2audio/models/qwen3-tts-voice-<чтец>-8bit
(lora_to_mlx_11a.py), кладёт в ~/.cache/text2audio/voices/qwen/ голос «<чтец> (обучен)»:
образец - тот же, что в обучении (вектор голоса), и «.model» с папкой модели
(core/engines/mlxaudio.py: trained_model) и «.gen.json» с выборкой. Только для личного использования: голоса
настоящих чтецов в git и в установщик не входят.

    python "Инструменты разработки/install_trained_11a.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import shutil

READERS = ("Игорь Ященко", "Вячеслав Герасимов")


def name_of(reader: str) -> str:
    return f"{reader} (обучен)"


def main() -> int:
    from core.engines.f5 import default_voices_dir
    from core.engines.mlxaudio import own_voices_dir, models_dir, trained_model
    d = own_voices_dir()
    _os.makedirs(d, exist_ok=True)
    for r in READERS:
        folder = f"qwen3-tts-voice-{r}-8bit"
        if not _os.path.exists(_os.path.join(models_dir(), folder, "config.json")):
            print(f"{r}: модели нет - пропуск")
            continue
        n = name_of(r)
        for ext in (".wav", ".txt"):
            shutil.copy(_os.path.join(default_voices_dir(), r + ext), _os.path.join(d, n + ext))
        open(_os.path.join(d, n + ".model"), "w", encoding="utf-8").write(folder + "\n")
        # с образцом и его расшифровкой (ICL) и выборкой спокойнее, чем у клона: замер 21.10 на
        # начале главы - слов 94 % (только вектор голоса - 88-90 %, клон - 95 %), похожесть 0,83
        json.dump({"icl": True, "temperature": 0.8, "top_p": 0.85, "top_k": 30},
                  open(_os.path.join(d, n + ".gen.json"), "w", encoding="utf-8"))
        print(f"{n} -> {trained_model(n)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
