#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, шаг 1: синтез Qwen ru-stress (8 бит) с ровным голосом.

Запускать окружением mlx-audio (под ../.heavy-lock):
    .venv-mlxaudio/bin/python "Инструменты разработки/voice_synth_39.py" РЕЖИМ ...

Режимы (20 фраз набора 3.8, Архив/голоса диктора 3.9/фразы.json):
  native:<Голос NN>        «голос диктора» самой ru-stress: ICL с его образцом
                           (Архив/голоса диктора 3.9/образцы/), одно зерно на все фразы
  icl_fixed:<чтец>         наш чтец, ICL с его образцом, одно зерно на все фразы
                           (многоразовый голосовой промпт: образец, расшифровка, зерно - одни)
  xvec_centroid:<чтец>     только вектор голоса - средний по ~60 фрагментам его аудиокниг
                           (вектор считает кодировщик голоса самой Qwen)
  icl_centroid:<чтец>      ICL с образцом + средний вектор голоса вместо вектора образца
  varseed:<чтец>           как в наборе 3.8: ICL с образцом, своё зерно на каждый кусок (база для сравнения)
  native105:<Голос NN>     как native, но repetition_penalty 1,05 (как у модели), а не 1,5 mlx-audio
Звук: Архив/голоса диктора 3.9/<режим>/<метка>__NN.wav
"""
import glob, json, os, sys, time
import numpy as np
import mlx.core as mx
import soundfile as sf
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "Архив", "голоса диктора 3.9")
RS = os.path.expanduser("~/.cache/text2audio/models/qwen3-tts-ru-stress-8bit")
VDIR = os.path.expanduser("~/.cache/text2audio/voices")
SEED = 7

# набор: «куски» (по умолчанию) - 17 трудных фраз в 5 кусков по 3-4 предложения, как
# режет программа, и 3 абзаца; «фразы» - 20 отдельных фраз (короткие у ICL часто пустые)
SETNAME = os.environ.get("VOICE_SET", "куски")
phr = json.load(open(os.path.join(BANK, SETNAME + ".json"), encoding="utf-8"))
model = load_model(RS)
orig_spk = model.extract_speaker_embedding


def centroid(voice: str):
    embs = []
    for f in sorted(glob.glob(os.path.join(BANK, "фрагменты", voice, "*.wav"))):
        a = load_audio(f, sample_rate=24000)
        embs.append(orig_spk(a[None] if a.ndim == 1 else a))
    c = mx.mean(mx.concatenate(embs, axis=0), axis=0, keepdims=True)
    mx.eval(c)
    return c, len(embs)


for spec in sys.argv[1:]:
    mode, label = spec.split(":", 1)
    if mode in ("native", "native105"):
        wav = os.path.join(BANK, "образцы", label + ".wav")
        ref_text = open(os.path.join(BANK, "образцы", label + ".txt"), encoding="utf-8").read().strip()
    else:
        wav = os.path.join(VDIR, label + ".wav")
        ref_text = open(os.path.join(VDIR, label + ".txt"), encoding="utf-8").read().replace("+", "").strip()
    ref = load_audio(wav, sample_rate=model.sample_rate)
    model.extract_speaker_embedding = orig_spk
    note = ""
    if "centroid" in mode:
        c, n = centroid(label)
        model.extract_speaker_embedding = lambda audio, sr=24000, _c=c: _c
        note = f", вектор по {n} фрагментам"
    out = os.path.join(BANK, mode if SETNAME == "фразы" else f"{mode} ({SETNAME})")
    os.makedirs(out, exist_ok=True)
    t0, audio_sec = time.time(), 0.0
    empty = 0
    silent = 0
    for i, p in enumerate(phr):
        a = np.zeros(0, np.float32)
        for att in range(3):                       # пустой выход - следующее зерно (до 2 раз)
            # одно зерно на все фразы; «varseed» - как раньше: своё зерно на каждый кусок
            mx.random.seed((38 + i if mode == "varseed" else SEED) + att)
            if mode == "xvec_centroid":
                gen = model.generate(text=p["acute"], ref_audio=ref, lang_code="russian")
            elif mode.endswith("105"):              # repetition_penalty как у модели (1,05)
                gen = model._generate_icl(text=p["acute"], ref_audio=ref, ref_text=ref_text,
                                          language="russian", repetition_penalty=1.05)
            else:
                gen = model.generate(text=p["acute"], ref_audio=ref, ref_text=ref_text,
                                     lang_code="russian")
            parts = [np.asarray(r.audio, np.float32).reshape(-1) for r in gen]
            a = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            if len(a) > model.sample_rate // 2:
                break
            empty += 1
        silent += len(a) <= model.sample_rate // 2
        audio_sec += len(a) / model.sample_rate
        sf.write(os.path.join(out, f"{label}__{i + 1:02d}.wav"), a, model.sample_rate)
    print(f"{spec}: {audio_sec:.0f} с звука за {time.time() - t0:.0f} с{note}; пустых выходов {empty}, фраз так и не озвучено {silent}",
          flush=True)
