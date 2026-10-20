#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Г: ударения F5 после части А - те же 8 кусков шага 1 (17 трудных
слов), голоса Ященко и Герасимов, разметка - нынешняя (по предложению, словари).
Тот же путь, что у книги: куски F5 по 187 знаков, проверка слов по звуку.
Пишет «Архив/голоса диктора 3.9/f5 11а (куски)/<чтец>__NN.wav»; замер - тем же
measure_bank_39.py (meter2 против верной разметки трудных слов). Под ../.heavy-lock.

    python "Инструменты разработки/f5_kuski_11a.py"
    VOICE_SET=куски python "Инструменты разработки/measure_bank_39.py" "f5 11а (куски)"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json

BANK = "Архив/голоса диктора 3.9"
OUT = f"{BANK}/f5 11а (куски)"


def main() -> int:
    import soundfile as sf
    import core.pipeline as P
    from core.engines import get_engine
    from core.profiles import PROFILES, engine_kwargs
    phr = json.load(open(f"{BANK}/куски.json", encoding="utf-8"))
    p = PROFILES["main"]
    eng = get_engine(p.engine, **engine_kwargs(p))
    eng.ensure_loaded()
    _os.makedirs(OUT, exist_ok=True)
    for voice in ("Вячеслав Герасимов", "Игорь Ященко"):
        for i, ph in enumerate(phr):
            body = P._prepare_text(ph["phrase"], "+", True, True, "ru")
            chunks = P._chunk_paragraphs(body, eng.max_chunk_chars, 0.4, 0.8, 1.1)
            au = P._synth_chunks(eng, voice, chunks, eng.sample_rate, 0.4, use_cache=False,
                                 sound_check=True)
            sf.write(f"{OUT}/{voice}__{i + 1:02d}.wav", au, eng.sample_rate)
            print(voice, i + 1, body[:80], flush=True)
    P.unload_engines()
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
