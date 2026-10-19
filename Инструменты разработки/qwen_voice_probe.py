#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Почему Qwen ru-stress не держит голос (3.8, после оценки Андрея 19.10).

Запускать окружением mlx-audio (под ../.heavy-lock):
    .venv-mlxaudio/bin/python "Инструменты разработки/qwen_voice_probe.py" [режимы]

Режимы (10 фраз × 2 чтеца, одно зерно на фразу во всех режимах):
  rs_now        ru-stress 8 бит, как сейчас: generate(), mlx-audio в режиме ICL
                поднимает repetition_penalty до 1,5
  rs_rp105      ru-stress, ICL с repetition_penalty 1,05 - как в generation_config
                модели (и в исходном qwen_tts)
  rs_rp105_t06  то же и temperature 0,6
  rs_xvec       ru-stress, только вектор голоса (без расшифровки образца, без ICL)
  base_now      исходная Qwen3-TTS-12Hz-1.7B-Base (8 бит) с тем же образцом, как сейчас
  base_rp105    исходная Base, repetition_penalty 1,05
  base_plain    исходная Base, текст без знаков ударения (она их не знает)
  lora05/075/10 исходная Base (bf16) + адаптер ru-stress с силой 0,5 / 0,75 / 1,0
Звук - Архив/qwen голос 3.8/<режим>/<чтец>__NN.wav.
"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
import soundfile as sf
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Архив", "qwen голос 3.8")
VOICES = ("Вячеслав Герасимов", "Игорь Ященко")
VDIR = os.path.expanduser("~/.cache/text2audio/voices")
RS = os.path.expanduser("~/.cache/text2audio/models/qwen3-tts-ru-stress-8bit")
BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
MODES = {"rs_now": (RS, "gen", 0.9, None), "rs_rp105": (RS, "icl", 0.9, 1.05),
         "rs_rp105_t06": (RS, "icl", 0.6, 1.05), "rs_xvec": (RS, "xvec", 0.9, 1.05),
         "base_now": (BASE, "gen", 0.9, None), "base_rp105": (BASE, "icl", 0.9, 1.05),
         "base_plain": (BASE, "gen", 0.9, None),       # Base без знаков ударения в тексте
         # Base (bf16) + адаптер ru-stress с силой s: W += s * alpha/r * B @ A
         "lora05": ("lora:0.5", "gen", 0.9, None), "lora075": ("lora:0.75", "gen", 0.9, None),
         "lora10": ("lora:1.0", "gen", 0.9, None)}
BASE16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"


def load_lora(scale: float):
    """Исходная Base + адаптер ru-stress (lora/adapter.safetensors) с силой scale."""
    from huggingface_hub import hf_hub_download
    m = load_model(BASE16)
    ad = mx.load(hf_hub_download("siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf",
                                 "lora/adapter.safetensors"))
    cfg = json.load(open(hf_hub_download("siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf",
                                         "lora/adapter.json")))
    k = scale * cfg["alpha"] / cfg["r"]
    from mlx.utils import tree_flatten, tree_unflatten
    params = dict(tree_flatten(m.parameters()))
    upd = {}
    for name in ad:
        if not name.endswith("lora_A.weight"):
            continue
        tgt = "talker.model." + name.replace(".lora_A.weight", ".weight")
        A, B = ad[name].astype(mx.float32), ad[name.replace("lora_A", "lora_B")].astype(mx.float32)
        W = params[tgt]
        upd[tgt] = (W.astype(mx.float32) + k * (B @ A)).astype(W.dtype)
    m.update(tree_unflatten(list(upd.items())))
    mx.eval(m.parameters())
    print(f"адаптер: {len(upd)} матриц, сила {scale}", flush=True)
    return m

phr = json.load(open(os.path.join(OUT, "фразы.json"), encoding="utf-8"))
want = sys.argv[1:] or list(MODES)
cur, model = None, None
for mode in want:
    path, how, temp, rp = MODES[mode]
    if path != cur:
        model, cur = (load_lora(float(path.split(":")[1])) if path.startswith("lora:")
                      else load_model(path)), path
    os.makedirs(os.path.join(OUT, mode), exist_ok=True)
    t0 = time.time()
    for v in VOICES:
        ref_text = open(os.path.join(VDIR, v + ".txt"), encoding="utf-8").read().replace("+", "").strip()
        ref = load_audio(os.path.join(VDIR, v + ".wav"), sample_rate=model.sample_rate)
        for i, p in enumerate(phr):
            mx.random.seed(38 + i)
            text = p["phrase"] if mode == "base_plain" else p["acute"]
            if how == "gen":
                gen = model.generate(text=text, ref_audio=ref, ref_text=ref_text, lang_code="russian",
                                     temperature=temp)
            elif how == "icl":
                gen = model._generate_icl(text=text, ref_audio=ref, ref_text=ref_text, language="russian",
                                          temperature=temp, repetition_penalty=rp)
            else:
                gen = model.generate(text=text, ref_audio=ref, lang_code="russian", temperature=temp,
                                     repetition_penalty=rp)
            a = np.concatenate([np.asarray(r.audio, np.float32).reshape(-1) for r in gen])
            sf.write(os.path.join(OUT, mode, f"{v}__{i + 1:02d}.wav"), a, model.sample_rate)
    print(f"{mode}: {time.time() - t0:.0f} с", flush=True)
