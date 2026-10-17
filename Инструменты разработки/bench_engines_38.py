#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Слепое сравнение движков 3.8 (задание 10а, п. 2).

Движки (в наборах - буквами): Qwen3-TTS ru-stress, F5 Misha (текущий), Chatterbox
Multilingual V3, VoxCPM2. Голоса - Герасимов и Ященко, те же образцы. Два
отдельных набора на голос:
  * «ударения» - 17 фраз (упрямые слова, омографы, редкие слова, ударная гласная
    в начале слова); слушать только ударение в слове из таблицы;
  * «звук» - 3 длинных абзаца; слушать только качество (металл, естественность).
Буквы в каждом наборе - свои, случайные; громкость выровнена. Ключи - в docs.
В ключ - и счёт meter2 (детектор ударения по звуку), чтобы сравнить со слухом.

Тяжёлый шаг - под ../.heavy-lock:
    python "Инструменты разработки/bench_engines_38.py" synth     # звук всех движков
    python "Инструменты разработки/bench_engines_38.py" build     # наборы, ключи, meter2
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import random
import re
import subprocess

import numpy as np

ENGINES = {"qwen": "Qwen3-TTS 1.7B ru-stress-cf (8 бит, mlx-audio)",
           "f5": "F5-TTS Misha (текущий движок, «Сбалансировано»)",
           "chatterbox": "Chatterbox Multilingual V3 (mlx-audio)",
           "voxcpm2": "VoxCPM2 (8 бит, mlx-audio, 48 кГц)"}
VOICES = ("Вячеслав Герасимов", "Игорь Ященко")
LETTERS = "БДНР"
OUT = "Архив/сравнение движков 3.8"
SEED = 38


def rows():
    out = []
    for l in open("Инструменты разработки/фразы_38.txt", encoding="utf-8"):
        if l.strip() and not l.startswith("#"):
            ph, w = l.rstrip("\n").rsplit("|", 1)
            out.append((ph, None if w == "-" else w))
    return out


def marked_text(phrase: str, word: str | None) -> str:
    """Фраза с ударениями RUAccent; у слова набора - ударение, как в наборе."""
    import core.pipeline as P
    t = P._with_end_punct(P._prepare_text(phrase, "+", True, True, "ru"))
    if word:
        plain = word.replace("+", "")
        pat = re.compile(r"(?<![\w+])" + r"\+?".join(map(re.escape, plain)) + r"(?![\w])",
                         re.IGNORECASE)
        t = pat.sub(word, t, count=1)
    return t


def wav_path(eng, voice, i):
    return _os.path.join(OUT, eng, f"{voice}__{i + 1:02d}.wav")


def synth(a) -> int:
    import soundfile as sf
    from core.engines import get_engine, clone_engine
    from core import profiles
    from core.engines.mlxaudio import MlxAudioEngine
    R = rows()
    texts = [marked_text(p, w) for p, w in R]
    for name in a.engines.split(","):
        _os.makedirs(_os.path.join(OUT, name), exist_ok=True)
        if name == "f5":
            import mlx.core as mx
            eng = get_engine(clone_engine(), language="ru",
                             **profiles.engine_kwargs(profiles.get(None), "balanced"))
        else:
            eng = MlxAudioEngine(model={"qwen": "qwen-stress"}.get(name, name), bits=8)
        eng.ensure_loaded()
        for voice in VOICES:
            for i, t in enumerate(texts):
                p = wav_path(name, voice, i)
                if _os.path.exists(p):
                    continue
                if name == "f5":
                    mx.random.seed(SEED + i)
                    au = eng.synth_batch([t], voice)[0]
                else:
                    au = eng.synth_chunk(t, voice, seed=SEED + i)
                sf.write(p, au, eng.sample_rate)
                print(f"{name} / {voice}: {i + 1}/{len(texts)}", flush=True)
        if hasattr(eng, "close"):
            eng.close()
        del eng
    return 0


def _level(a: np.ndarray) -> np.ndarray:
    """Одна громкость у всех вариантов: RMS речи -> 0,07, пики не выше 0,95."""
    a = np.asarray(a, np.float32)
    voiced = a[np.abs(a) > 0.01]
    rms = float(np.sqrt(np.mean(voiced ** 2))) if len(voiced) else 0.0
    if rms > 0:
        a = a * (0.07 / rms)
    pk = float(np.abs(a).max()) if len(a) else 0.0
    return a * (0.95 / pk) if pk > 0.95 else a


def build(a) -> int:
    import soundfile as sf
    from core import meter
    from core.stress import to_combining
    R = rows()
    stress_idx = [i for i, (_p, w) in enumerate(R) if w]
    long_idx = [i for i, (_p, w) in enumerate(R) if not w]
    names = [n for n in ENGINES if _os.path.exists(_os.path.join(OUT, n))]
    auto = {}
    for name in names:
        for voice in VOICES:
            ok = n = 0
            wrong = []
            for i in stress_idx:
                p = wav_path(name, voice, i)
                if not _os.path.exists(p):
                    continue
                au, sr = sf.read(p, dtype="float32")
                ph, w = R[i]
                bad = meter.wrong_words(au, sr, marked_text(ph, w), only={w.replace("+", "").lower()},
                                        min_margin=0.0)
                n += 1
                ok += not bad
                if bad:
                    wrong.append(w)
            auto[(name, voice)] = (ok, n, wrong)
            print(f"meter2: {name} / {voice}: верно {ok} из {n}", flush=True)
    for voice in VOICES:
        for kind, idx in (("ударения", stress_idx), ("звук", long_idx)):
            rnd = random.Random(f"3.8 {voice} {kind}")
            letters = list(LETTERS[:len(names)])
            rnd.shuffle(letters)
            mapping = dict(zip(names, letters))
            folder = f"Сравнение 3.8 - движки, {kind} - {voice}"
            _os.makedirs(folder, exist_ok=True)
            for name, L in mapping.items():
                parts, sr0 = [], None
                for i in idx:
                    au, sr = sf.read(wav_path(name, voice, i), dtype="float32")
                    sr0 = sr0 or sr
                    parts += [_level(au), np.zeros(int(1.5 * sr), np.float32)]
                tmp = _os.path.join(OUT, f"{voice}__{kind}__{name}.wav")
                sf.write(tmp, np.concatenate(parts), sr0)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ar", "48000",
                                "-b:a", "256k", _os.path.join(folder, f"Вариант {L}.mp3")], check=True)
            Ls = sorted(mapping.values())
            if kind == "ударения":
                head = "| № | фраза | слово (как надо) | " + " | ".join(f"{L}: верно?" for L in Ls) + " |"
                tab = [head, "|" + "---|" * (3 + len(Ls))]
                for k, i in enumerate(idx):
                    ph, w = R[i]
                    tab.append(f"| {k + 1} | {ph} | {to_combining(w.lower())} |" + " |" * len(Ls))
                intro = ("В каждом файле - одни и те же 17 фраз (пауза 1,5 с), варианты под буквами. "
                         "Слушать **только ударение** в слове из третьей колонки (да / нет).")
            else:
                tab = ["| вариант | металл, призвук (0 нет - 3 сильно) | естественность (1-5) | заметки |",
                       "|---|---|---|---|"] + [f"| {L} | | | |" for L in Ls]
                intro = ("В каждом файле - одни и те же 3 длинных абзаца. Ударения здесь не судим - "
                         "слушать **только звук**: металл, шипение, естественность, ровность голоса.")
            open(_os.path.join(folder, "Оценка.md"), "w", encoding="utf-8").write(
                f"# Движки 3.8, {kind}: {voice}\n\n{intro}\n\nКлюч - в docs, открывать после "
                "оценки.\n\n" + "\n".join(tab) + "\n")
            key = [f"# Ключ 3.8 - движки, {kind}, {voice}", ""]
            key += [f"- **Вариант {mapping[n]}** - {ENGINES[n]}" for n in names]
            if kind == "ударения":
                key += ["", "Счёт meter2 (детектор ударения по звуку), до вашей оценки:", ""]
                for n in names:
                    ok, cnt, wrong = auto.get((n, voice), (0, 0, []))
                    key.append(f"- {mapping[n]}: верно {ok} из {cnt}" +
                               (f"; не так: {', '.join(to_combining(w.lower()) for w in wrong)}" if wrong else ""))
            open(f"docs/ключ 3.8 - движки, {kind}, {voice} (не открывать до оценки).md", "w",
                 encoding="utf-8").write("\n".join(key) + "\n")
    json.dump({f"{n}|{v}": list(x) for (n, v), x in auto.items()},
              open(_os.path.join(OUT, "meter2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["synth", "build"])
    ap.add_argument("--engines", default=",".join(ENGINES))
    a = ap.parse_args()
    return synth(a) if a.stage == "synth" else build(a)


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
