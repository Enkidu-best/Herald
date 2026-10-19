#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, шаг 2: хватает ли этого Mac для дообучения Qwen3-TTS на голосе чтеца.

Запускать окружением .venv-train (PyTorch + MPS, qwen-tts, peft), под ../.heavy-lock:
    .venv-train/bin/python "Инструменты разработки/train_check_39.py" --voice "Игорь Ященко"

Что делает - НАСТОЯЩИЕ шаги обучения, не оценку на бумаге:
  1. 16 фраз чтеца из ~/.cache/text2audio/train/<чтец>/train_raw.jsonl (prepare_voice_data_39.py);
  2. коды звука - официальным токенизатором Qwen3-TTS-Tokenizer-12Hz;
  3. модель - Qwen3-TTS ru-stress (исходные веса PyTorch), bf16, на видеоядре (MPS);
  4. LoRA r=16 (alpha 32) на внимании и MLP «talker» - как адаптер ru-stress;
  5. потеря - как в официальном finetuning/sft_12hz.py (QwenLM/Qwen3-TTS):
     codec_0 + 0,3 * sub_talker; голос - вектор из ref_audio;
  6. замер: секунд на шаг (пакет 1 и 2), память видеоядра; пересчёт на полное
     обучение (фраз в наборе × эпох).
Полное обучение всех весов (как в официальном скрипте) на 16 ГБ не помещается:
AdamW держит две копии весов + градиенты (1,7 млрд × 2 байта × 4 ≈ 14 ГБ без
активаций) - поэтому на Mac проверяется только LoRA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "qwen_ft"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Игорь Ященко")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--init", default="siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    import torch
    from transformers import AutoConfig
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from peft import LoraConfig, inject_adapter_in_model
    from dataset import TTSDataset

    d = os.path.expanduser(f"~/.cache/text2audio/train/{a.voice}")
    rows = [json.loads(l) for l in open(os.path.join(d, "train_raw.jsonl"), encoding="utf-8")]
    n_all, h_all = len(rows), sum(r["dur"] for r in rows) / 3600
    rows = sorted(rows, key=lambda r: r["dur"])[len(rows) // 2 - a.n // 2: len(rows) // 2 + a.n // 2]
    dev = a.device

    t0 = time.time()
    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map=dev)
    enc = tok.encode([r["audio"] for r in rows])
    for r, c in zip(rows, enc.audio_codes):
        r["audio_codes"] = c.cpu().tolist()
    t_codes = time.time() - t0
    del tok
    print(f"коды звука: {len(rows)} фраз за {t_codes:.1f} с", flush=True)

    t0 = time.time()
    m = Qwen3TTSModel.from_pretrained(a.init, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                      device_map=dev)
    print(f"модель загружена за {time.time() - t0:.0f} с", flush=True)
    model = m.model
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                     target_modules=r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
    inject_adapter_in_model(cfg, model.talker.model)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA: {sum(p.numel() for p in params) / 1e6:.1f} млн обучаемых весов", flush=True)
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.01)
    config = AutoConfig.from_pretrained(a.init)
    ds = TTSDataset(rows, m.processor, config)
    model.train()

    def step(batch):
        input_ids = batch["input_ids"].to(dev)
        codec_ids = batch["codec_ids"].to(dev)
        ref_mels = batch["ref_mels"].to(dev)
        tem = batch["text_embedding_mask"].to(dev)
        cem = batch["codec_embedding_mask"].to(dev)
        am = batch["attention_mask"].to(dev)
        lab = batch["codec_0_labels"].to(dev)
        cmask = batch["codec_mask"].to(dev)
        with torch.no_grad():
            spk = model.speaker_encoder(ref_mels.to(model.dtype))
        te = model.talker.model.text_embedding(input_ids[:, :, 0]) * tem
        ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * cem
        ce[:, 6, :] = spk
        emb = te + ce
        for i in range(1, 16):
            emb = emb + model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) \
                * cmask.unsqueeze(-1)
        out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=am[:, :-1], labels=lab[:, 1:],
                           output_hidden_states=True)
        hs = out.hidden_states[0][-1][cmask[:, :-1]]
        _l, sub = model.talker.forward_sub_talker_finetune(codec_ids[cmask], hs)
        loss = out.loss + 0.3 * sub
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        return float(loss.item())

    res = {"фраз в наборе": n_all, "часов в наборе": round(h_all, 2)}
    for bs in (1, 2):
        times = []
        for k in range(a.steps):
            items = [ds[(k * bs + j) % len(ds)] for j in range(bs)]
            batch = ds.collate_fn(items)
            torch.mps.synchronize() if dev == "mps" else None
            t0 = time.time()
            loss = step(batch)
            torch.mps.synchronize() if dev == "mps" else None
            times.append(time.time() - t0)
            print(f"пакет {bs}, шаг {k + 1}: {times[-1]:.1f} с, потеря {loss:.3f}", flush=True)
        sec = sorted(times[1:])[len(times[1:]) // 2]          # без первого (прогрев)
        res[f"с на шаг, пакет {bs}"] = round(sec, 2)
        res[f"с на фразу, пакет {bs}"] = round(sec / bs, 2)
    if dev == "mps":
        res["память видеоядра, ГБ"] = round(torch.mps.driver_allocated_memory() / 1e9, 1)
    per = min(res["с на фразу, пакет 1"], res["с на фразу, пакет 2"])
    for ep in (3, 5, 10):
        res[f"часов на {ep} эпох"] = round(per * n_all * ep / 3600, 1)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    json.dump(res, open(os.path.join(d, "проверка обучения на Mac.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
