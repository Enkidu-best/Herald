#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Д: обученный адаптер голоса -> модель MLX для Herald.

Запускать окружением mlx-audio, под ../.heavy-lock:
    .venv-mlxaudio/bin/python "Инструменты разработки/lora_to_mlx_11a.py" --voice "Игорь Ященко" [--epoch 6]

1. Веса ru-stress (PyTorch, bf16) + адаптер чтеца: W += alpha/r * B @ A на внимании и
   MLP «talker» (адаптер - train_lora_11a.py, ~/.cache/text2audio/train/<чтец>/lora/;
   эпоха - лучшая по потере на 30 отложенных фразах).
2. Слитые веса - во временную папку рядом с остальными файлами ru-stress.
3. mlx_audio.convert, 8 бит -> ~/.cache/text2audio/models/qwen3-tts-voice-<чтец>-8bit.
Модель - только для личного использования (голос чтеца): в git и в установщик не идёт.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"
KEEP = ("config.json", "generation_config.json", "merges.txt", "preprocessor_config.json",
        "tokenizer_config.json", "vocab.json", "speech_tokenizer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Игорь Ященко")
    ap.add_argument("--epoch", type=int, default=0, help="0 - лучшая по потере на отложенных фразах")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--suffix", default="", help="к имени папки модели (пробы эпох)")
    ap.add_argument("--lora", default="lora", help="папка адаптера в папке чтеца")
    ap.add_argument("--base", default=REPO, help="основа: ru-stress (11а) или Qwen/Qwen3-TTS-12Hz-1.7B-Base (11б)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="сила адаптера: 0,5-0,75 - надёжнее чтение основы, чуть меньше похожесть")
    a = ap.parse_args()
    import mlx.core as mx

    lora = os.path.expanduser(f"~/.cache/text2audio/train/{a.voice}/{a.lora}")
    if not a.epoch:                    # лучшая эпоха - по строкам «эпоха N: … на отложенных X» журнала
        held = {int(m.group(1)): float(m.group(2)) for m in re.finditer(
            r"эпоха (\d+): потеря обучения [\d.]+, на отложенных ([\d.]+)",
            open(os.path.join(lora, "train.log"), encoding="utf-8").read())}
        a.epoch = min(held, key=held.get)
        print(f"{a.voice}: потеря на отложенных по эпохам {held} -> эпоха {a.epoch}", flush=True)
    ad_path = os.path.join(lora, f"epoch_{a.epoch:02d}.safetensors")
    cfg = json.load(open(os.path.join(lora, "adapter.json"), encoding="utf-8"))
    models = os.path.expanduser("~/.cache/text2audio/models")
    dst = os.path.join(models, f"qwen3-tts-voice-{a.voice}-{a.bits}bit{a.suffix}")
    tmp = os.path.join(models, f"_merge-{a.voice}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    t0 = time.time()
    # папка уже скачанной ru-stress (в кэше нет README и мелочей - snapshot_download без сети отказывает)
    src = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--" + a.base.replace("/", "--") + "/snapshots/*"))[0]
    for name in KEEP:
        os.symlink(os.path.realpath(os.path.join(src, name)), os.path.join(tmp, name))
    W = mx.load(os.path.join(src, "model.safetensors"))
    ad = mx.load(ad_path)
    k = a.scale * cfg["alpha"] / cfg["r"]
    n = 0
    for name in ad:
        if not name.endswith("lora_A.weight"):
            continue
        tgt = "talker.model." + name.replace(".lora_A.weight", ".weight")
        A, B = ad[name].astype(mx.float32), ad[name.replace("lora_A", "lora_B")].astype(mx.float32)
        W[tgt] = (W[tgt].astype(mx.float32) + k * (B @ A)).astype(W[tgt].dtype)
        n += 1
    mx.eval(W)
    mx.save_safetensors(os.path.join(tmp, "model.safetensors"), W, metadata={"format": "pt"})
    del W
    print(f"{a.voice}: влито {n} матриц из {os.path.basename(ad_path)} за {time.time() - t0:.0f} с", flush=True)

    shutil.rmtree(dst, ignore_errors=True)
    cmd = [sys.executable, "-m", "mlx_audio.convert", "--hf-path", tmp, "--mlx-path", dst,
           "-q", "--q-bits", str(a.bits), "--model-domain", "tts"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode or not os.path.exists(os.path.join(dst, "config.json")):
        print((r.stderr or r.stdout)[-3000:])
        return 1
    json.dump({"voice": a.voice, "adapter": os.path.basename(ad_path), "epoch": a.epoch, "lora": a.lora,
               "scale": a.scale, "base": a.base,
               "personal_use_only": True},
              open(os.path.join(dst, "herald_voice.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"готово: {dst} за {time.time() - t0:.0f} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
