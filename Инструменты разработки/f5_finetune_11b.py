#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б, шаг 3б: ПРОБНОЕ дообучение F5 (Misha accent_tune) на фразах чтеца.

Модель - официальная DiT из f5-tts с архитектурой F5TTS_v1_Base (как DIT_ARCH движка Herald
f5torch), веса Misha24-10/F5-TTS_RUSSIAN accent_tune (EMA). Цель обучения - та же, что у F5
(CFM: восстановить замаскированную часть мел-спектра по тексту и остальному звуку).
LoRA r=32 на внимании и feed-forward блоков DiT (полное обучение с AdamW на 16 ГБ Mac - на
грани: ~5,5 ГБ только веса и состояния). Текст - разметка конвейера с
«+» (как F5 получает при озвучке). Фразы - очищенные (11а Д) и, по шагу 2, «восстановленные»
DeepFilterNet (Sidon меняет голос - похожесть -0,12...-0,2).

    .venv/bin/python "Инструменты разработки/f5_finetune_11b.py" --voice "Игорь Ященко" --probe 20
    .venv/bin/python "Инструменты разработки/f5_finetune_11b.py" --voice "Игорь Ященко" --minutes 45
После обучения LoRA вливается в веса: <чтец>/f5-lora/model_v1.safetensors (+ vocab.txt) -
их берёт движок f5torch через переменную HERALD_F5_WEIGHTS (только для проб).
Голос чтеца - только для личного использования.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
TARGET = r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.ff\.(0\.0|2))"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Игорь Ященко")
    ap.add_argument("--data", default="train_clean_dfn.jsonl")
    ap.add_argument("--probe", type=int, default=0, help="только N шагов - замер скорости и памяти")
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--frames", type=int, default=900, help="кадров мела в пакете (~9,6 с звука)")
    ap.add_argument("--epochs", type=int, default=0, help="полное обучение: эпох (0 - по времени --minutes)")
    ap.add_argument("--hold", type=int, default=30, help="фраз отложить для проверки после каждой эпохи")
    ap.add_argument("--out", default="f5-lora", help="папка в папке чтеца")
    a = ap.parse_args()
    import torch
    import soundfile as sf
    from safetensors.torch import load_file, save_file
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, inject_adapter_in_model
    from f5_tts.model import CFM, DiT
    from f5_tts.model.modules import MelSpec
    import core.pipeline as P
    from core.engines.f5_torch import DIT_ARCH
    from core.engines.f5_mlx import LANG_MODELS

    dev = "mps"
    d = os.path.expanduser(f"~/.cache/text2audio/train/{a.voice}")
    out = os.path.join(d, a.out)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "train.log"), "a", encoding="utf-8")

    def say(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    cfg = LANG_MODELS["ru"]
    ckpt, vocab_path = hf_hub_download(cfg["repo"], cfg["ckpt"]), hf_hub_download(cfg["repo"], cfg["vocab"])
    chars = [l[:-1] if l.endswith("\n") else l for l in open(vocab_path, encoding="utf-8")]
    vocab = {c: i for i, c in enumerate(chars)}
    sd = load_file(ckpt)
    pre = "ema_model.transformer."
    tsd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    arch = {k: v for k, v in DIT_ARCH.items() if k not in ("attn_backend", "attn_mask_enabled")}
    n_text = tsd["text_embed.text_embed.weight"].shape[0] - 1      # как в движке f5torch
    # пересчёт активаций у DiT - reentrant: при замороженных входах градиент через него не идёт
    net = DiT(**arch, text_num_embeds=n_text, checkpoint_activations=False)
    miss, _unexp = net.load_state_dict(tsd, strict=False)
    if miss:
        raise RuntimeError(f"нет слоёв: {miss[:5]}")
    model = CFM(transformer=net, mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024,
                                                      n_mel_channels=100, target_sample_rate=24000,
                                                      mel_spec_type="vocos"),
                vocab_char_map=vocab).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=TARGET), net)
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    say(f"{a.voice}: F5 Misha + LoRA {sum(p.numel() for p in params) / 1e6:.1f} млн обучаемых")
    mel_fn = MelSpec(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                     target_sample_rate=24000, mel_spec_type="vocos")

    rows = [json.loads(l) for l in open(os.path.join(d, a.data), encoding="utf-8")]
    items = []
    for r in rows:
        w, sr = sf.read(r["audio"], dtype="float32")
        mel = mel_fn(torch.from_numpy(w)[None])[0].T            # (кадры, 100)
        if mel.shape[0] > a.frames:     # длиннее пакета: внимание на такой длине не помещается в память
            continue
        text = P._prepare_text(r["text_plain"], "+", True, True, "ru")
        items.append((mel, list(text)))
    rnd = random.Random(11)
    hold = []
    if a.epochs:
        order = list(range(len(items)))
        rnd.shuffle(order)
        hold = [items[i] for i in order[:a.hold]]
        items = [items[i] for i in order[a.hold:]]
    say(f"фраз {len(items)} + {len(hold)} отложено, {sum(m.shape[0] for m, _ in items) * 256 / 24000 / 3600:.2f} ч")

    def batches():
        idx = list(range(len(items)))
        rnd.shuffle(idx)
        cur, n = [], 0
        for i in idx:
            f = items[i][0].shape[0]
            if cur and n + f > a.frames:
                yield cur
                cur, n = [], 0
            cur.append(i)
            n += f
        if cur:
            yield cur

    def held_loss() -> float:
        """Потеря на отложенных фразах при ОДИНАКОВОМ шуме и маске (зерно на фразу):
        иначе случайность CFM больше разницы между эпохами."""
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for k, (mel, text) in enumerate(hold):
                torch.manual_seed(1000 + k)
                l, _c, _p = model(mel[None].to(dev), text=[text], lens=torch.tensor([mel.shape[0]]).to(dev))
                tot += float(l.item())
        model.train()
        return tot / max(len(hold), 1)

    def lora_state():
        return {n: p.detach().to("cpu").clone() for n, p in named}

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    torch.set_grad_enabled(True)        # разметка ударений (модели Herald) выключает градиенты глобально
    say(f"градиенты: {torch.is_grad_enabled()}")
    model.train()
    t_end = time.time() + (a.minutes * 60 if not (a.probe or a.epochs) else 1e9)
    step, run, t0 = 0, 0.0, time.time()
    held = {}
    if a.epochs:
        held[0] = held_loss()
        say(f"эпоха 0: потеря на отложенных {held[0]:.4f}")
    ep = 0
    while time.time() < t_end and (not a.probe or step < a.probe) and (not a.epochs or ep < a.epochs):
        ep += 1
        for b in batches():
            mels = [items[i][0] for i in b]
            lens = torch.tensor([m.shape[0] for m in mels])
            inp = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True).to(dev)
            loss, _c, _p = model(inp, text=[items[i][1] for i in b], lens=lens.to(dev))
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            run += float(loss.item())
            if step % 25 == 0 or (a.probe and step <= 5):
                torch.mps.empty_cache()
                say(f"шаг {step}: потеря {run / step:.4f}, {(time.time() - t0) / step:.2f} с/шаг, "
                    f"видеоядро {torch.mps.driver_allocated_memory() / 1e9:.1f} ГБ")
            if (a.probe and step >= a.probe) or time.time() >= t_end:
                break
        if a.epochs:
            save_file(lora_state(), os.path.join(out, f"epoch_{ep:02d}.safetensors"))
            held[ep] = held_loss()
            say(f"эпоха {ep}: потеря на отложенных {held[ep]:.4f}, шагов {step}, {(time.time() - t0) / 60:.0f} мин")
    say(f"шагов {step}, {(time.time() - t0) / max(step, 1):.2f} с/шаг")
    if a.probe:
        return 0
    if a.epochs:                         # лучшая эпоха по отложенным фразам
        best = min((e for e in held if e), key=held.get)
        say(f"потеря на отложенных по эпохам {json.dumps({k: round(v, 4) for k, v in held.items()})} -> эпоха {best}")
        sd_best = load_file(os.path.join(out, f"epoch_{best:02d}.safetensors"))
        with torch.no_grad():
            for n, p in named:
                p.copy_(sd_best[n].to(p.device))
    # влить LoRA: W += alpha/r * B @ A; сохранить как веса движка (префикс EMA)
    state = {k: v.detach().to("cpu") for k, v in net.state_dict().items() if "lora_" not in k}
    merged = {}
    for n, mod in net.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A, B = mod.lora_A["default"].weight, mod.lora_B["default"].weight
            k = n + ".base_layer.weight"
            W = state.pop(k) if k in state else state.pop(n + ".weight")
            merged[n + ".weight"] = (W + (32 / 32) * (B @ A).to("cpu")).contiguous()
            if n + ".base_layer.bias" in state:
                merged[n + ".bias"] = state.pop(n + ".base_layer.bias")
    state.update(merged)
    # поверх ПОЛНОГО файла Misha (те же ключи и типы, что ждут движки f5mlx / f5torch);
    # меняются только веса трансформера EMA
    full = {k: v.contiguous() for k, v in sd.items()}
    for k, v in state.items():
        key = pre + k.replace(".base_layer", "")
        if key in full:
            full[key] = v.to(full[key].dtype).contiguous()
    save_file(full, os.path.join(out, "model_v1.safetensors"))
    import shutil
    shutil.copy(vocab_path, os.path.join(out, "vocab.txt"))
    json.dump({"voice": a.voice, "steps": step, "lr": a.lr, "data": a.data, "epochs": a.epochs,
               "held_loss": {str(k): round(v, 4) for k, v in held.items()},
               "best_epoch": (min((e for e in held if e), key=held.get) if a.epochs else None),
               "personal_use_only": True},
              open(os.path.join(out, "adapter.json"), "w", encoding="utf-8"), ensure_ascii=False)
    say("готово: " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
