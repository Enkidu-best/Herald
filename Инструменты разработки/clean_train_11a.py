#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Д: чистка набора для обучения голоса - текст должен совпадать со звуком.

Первое обучение (20.10) читало чисто голосом чтеца, но теряло ~10 % слов (числа,
имена, хвост фразы). Частая причина у таких моделей - пары «звук - текст», где текст
не совпадает со звуком (разрез Whisper по длинному файлу съел или прихватил слово):
модель учится, что часть текста можно не читать. Здесь каждая фраза распознаётся
заново, отдельно (Whisper large-v3-turbo), и сверяется с текстом набора:
  * расхождение букв (CER) больше 5 % - фраза уходит;
  * не совпало первое или последнее слово (обрезанный край) - уходит.
Текст оставшихся фраз - ровно в той форме, в какой его получает модель при озвучке
(разметка конвейера + plus_to_acute: без знака на односложных словах; в первом наборе
знак стоял и на них - «не́», «о́н»). Пишет <чтец>/train_clean.jsonl и <чтец>/чистка.json
(что ушло и почему). Под ../.heavy-lock.

    python "Инструменты разработки/clean_train_11a.py" "Игорь Ященко" "Вячеслав Герасимов"
    python "Инструменты разработки/clean_train_11a.py" "Вячеслав Герасимов" --src=train_olesya_raw.jsonl --dst=train_olesya_clean.jsonl
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import difflib
import json
import re

CER_MAX = 0.05            # как в больших наборах (отчёт: порог ~5 %)


def words(s: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in re.findall(r"[А-Яа-яЁёA-Za-z]+", s)]


def cer(a: str, b: str) -> float:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1 - sum(x.size for x in sm.get_matching_blocks()) / max(len(a), 1)


def main() -> int:
    import soundfile as sf
    import librosa
    import core.pipeline as P
    from core.engines.mlxaudio import plus_to_acute
    args = [x for x in _sys.argv[1:] if not x.startswith("--")]
    opt = dict(x[2:].split("=", 1) for x in _sys.argv[1:] if x.startswith("--") and "=" in x)
    src, dst = opt.get("src", "train_raw.jsonl"), opt.get("dst", "train_clean.jsonl")
    for voice in args:
        d = _os.path.expanduser(f"~/.cache/text2audio/train/{voice}")
        rows = [json.loads(l) for l in open(f"{d}/{src}", encoding="utf-8")]
        keep, gone, allc = [], [], []
        for i, r in enumerate(rows):
            a, sr = sf.read(r["audio"], dtype="float32")
            a16 = librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a
            heard = " ".join(w for w, _s, _e in (P._listen(a16, 16000, "ru", P._ASR_FULL) or []))
            tw, hw = words(r["text_plain"]), words(heard)
            c = cer(" ".join(tw), " ".join(hw))
            edge = bool(tw and hw and (tw[0] != hw[0] or tw[-1] != hw[-1]))
            allc.append({"audio": _os.path.basename(r["audio"]), "CER": round(c, 3), "край": edge, "слышно": heard})
            why = []
            if c > CER_MAX:
                why.append(f"CER {c:.2f}")
            if edge:
                why.append("край")
            if not hw:
                why.append("не распознано")
            if why:
                gone.append({"audio": _os.path.basename(r["audio"]), "почему": why, "текст": r["text_plain"],
                             "слышно": heard})
            else:
                r = dict(r, text=plus_to_acute(P._prepare_text(r["text_plain"], "+", True, True, "ru")))
                keep.append(r)
            if (i + 1) % 100 == 0:
                print(f"{voice}: {i + 1}/{len(rows)}, ушло {len(gone)}", flush=True)
        with open(f"{d}/{dst}", "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        res = {"было": len(rows), "осталось": len(keep), "ушло": len(gone),
               "часов осталось": round(sum(r["dur"] for r in keep) / 3600, 2), "фразы": gone,
               "все": allc}
        json.dump(res, open(f"{d}/чистка{'' if dst == 'train_clean.jsonl' else ' ' + dst[:-6]}.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{voice}: было {len(rows)}, осталось {len(keep)} ({res['часов осталось']} ч), ушло {len(gone)}",
              flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
