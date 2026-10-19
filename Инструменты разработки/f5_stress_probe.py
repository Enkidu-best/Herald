#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F5 (наш голос) + верные ударения (3.8, после оценки Андрея 19.10): опыт на 10 фразах.

Находка части 1: F5 не видит «+», когда ударная гласная - первая буква слова
(«+отмель», «+устье»). Режимы (одно зерно на фразу):
  f5_dbl        у таких слов гласная удваивается, «+» - внутри: «о+отмель»
  f5_hard       перед словом твёрдый знак: «ъ+отмель»
  f5_check      как сейчас, но ударение проверяет meter2: не там - другое зерно, до 2 раз
  f5_dbl_check  «о+отмель» и проверка meter2
Звук - Архив/qwen голос 3.8/<режим>/<чтец>__NN.wav. Под ../.heavy-lock.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import re

OUT = "Архив/qwen голос 3.8"
VOICES = ("Вячеслав Герасимов", "Игорь Ященко")
VOW = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def initial_fix(text: str, how: str) -> str:
    """«+отмель» -> «о+отмель» (dbl) или «ъ+отмель» (hard) - только где «+» в начале слова."""
    def sub(m):
        v, rest = m.group(2), m.group(3)
        if sum(c in VOW for c in rest) < 1:          # односложное («+он») - не трогаем
            return m.group(0)
        return m.group(1) + (v.lower() + "+" + v if how == "dbl" else "ъ+" + v) + rest
    return re.sub(r"(^|[\s«(\"—-])\+([" + VOW + r"])([А-Яа-яЁё]*)", sub, text)


def main() -> int:
    import mlx.core as mx
    import soundfile as sf
    from core import meter
    from core.engines import get_engine, clone_engine
    from core import profiles
    phr = json.load(open(_os.path.join(OUT, "фразы.json"), encoding="utf-8"))
    eng = get_engine(clone_engine(), language="ru",
                     **profiles.engine_kwargs(profiles.get(None), "balanced"))
    eng.ensure_loaded()
    modes = _sys.argv[1:] or ["f5_dbl", "f5_hard", "f5_check", "f5_dbl_check"]
    stats = {}
    for mode in modes:
        _os.makedirs(_os.path.join(OUT, mode), exist_ok=True)
        tries_total = 0
        for v in VOICES:
            eng.synth_batch(["Пров+ерка."], v)
            for i, p in enumerate(phr):
                text = p["marked"]
                if "dbl" in mode:
                    text = initial_fix(text, "dbl")
                elif "hard" in mode:
                    text = initial_fix(text, "hard")
                word = (p["word"] or "").replace("+", "").lower()
                best = None
                for att in range(3 if "check" in mode else 1):
                    mx.random.seed(38 + i + 1000 * att)
                    au = eng.synth_batch([text], v)[0]
                    bad = meter.wrong_words(au, eng.sample_rate, p["marked"], only={word},
                                            min_margin=0.0) if word else []
                    tries_total += att > 0
                    if best is None or not bad:
                        best = au
                    if not bad:
                        break
                sf.write(_os.path.join(OUT, mode, f"{v}__{i + 1:02d}.wav"), best, eng.sample_rate)
        stats[mode] = tries_total
        print(f"{mode}: пересчётов {tries_total}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
