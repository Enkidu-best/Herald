#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Д (шаг 2): обучение LoRA голоса чтеца поверх Qwen3-TTS ru-stress.

Запускать окружением .venv-train (PyTorch + MPS, qwen-tts, peft), под ../.heavy-lock:
    .venv-train/bin/python "Инструменты разработки/train_lora_11a.py" --voice "Игорь Ященко"

Тот же шаг, что в train_check_39.py (там - замер скорости), но на всём наборе:
  * фразы - ~/.cache/text2audio/train/<чтец>/train_raw.jsonl (prepare_voice_data_39.py),
    30 фраз отложены для проверки (потеря на них - после каждой эпохи);
  * коды звука - официальным токенизатором Qwen3-TTS-Tokenizer-12Hz, один раз
    (codes.jsonl рядом с набором);
  * LoRA r=16 (alpha 32) на внимании и MLP «talker», как адаптер ru-stress;
    потеря codec_0 + 0,3 * sub_talker (как в официальном finetuning/sft_12hz.py), но
    запрос - в той форме, в какой Herald озвучивает (см. loss_of: у официального
    скрипта две ошибки), голос - вектор из ref_audio (без ICL: образец не читается);
  * пакет 1 (без пустого места), накопление 8 (как пакет 2 × 4 в sft_12hz.py), AdamW;
    память: пересчёт активаций (gradient checkpointing), кэш видеоядра чистится каждые
    25 фраз, потолок - PYTORCH_MPS_HIGH_WATERMARK_RATIO (20.10: без этого обучение
    выросло до 15 ГБ и Mac ушёл в подкачку - рядом серверы Interview Coach и /speak);
  * после каждой эпохи адаптер - <чтец>/lora/epoch_NN.safetensors + adapter.json
    (имена как у адаптера ru-stress: layers.N.…lora_A.weight относительно talker.model).
Только для личного использования: голос чтеца не публикуется и в git не идёт.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")   # ~6,3 ГБ из 12,7; до импорта torch
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")    # не выше верхнего
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "qwen_ft"))
TARGET = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Игорь Ященко")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--init", default="siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probe", action="store_true", help="только потеря исходной модели")
    ap.add_argument("--data", default="train_raw.jsonl", help="набор фраз (train_clean.jsonl - после чистки)")
    ap.add_argument("--out", default="lora", help="папка адаптера в папке чтеца")
    a = ap.parse_args()
    import torch
    from transformers import AutoConfig
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import save_file
    from dataset import TTSDataset

    d = os.path.expanduser(f"~/.cache/text2audio/train/{a.voice}")
    out = os.path.join(d, a.out)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "train.log"), "a", encoding="utf-8")

    def say(msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    rows = [json.loads(l) for l in open(os.path.join(d, a.data), encoding="utf-8")]
    dev = a.device

    cpath = os.path.join(d, "codes.jsonl")
    codes = {}
    if os.path.exists(cpath):
        for l in open(cpath, encoding="utf-8"):
            r = json.loads(l)
            codes[r["audio"]] = r["codes"]
    todo = [r for r in rows if r["audio"] not in codes]
    if todo:
        t0 = time.time()
        tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map=dev)
        with open(cpath, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), 8):
                part = todo[i:i + 8]
                enc = tok.encode([r["audio"] for r in part])
                for r, c in zip(part, enc.audio_codes):
                    codes[r["audio"]] = c.cpu().tolist()
                    f.write(json.dumps({"audio": r["audio"], "codes": codes[r["audio"]]}) + "\n")
                if dev == "mps":
                    torch.mps.empty_cache()
        del tok
        if dev == "mps":
            torch.mps.empty_cache()
        say(f"коды звука: {len(todo)} фраз за {time.time() - t0:.0f} с")
    for r in rows:
        r["audio_codes"] = codes[r["audio"]]

    rnd = random.Random(11)
    order = list(range(len(rows)))
    rnd.shuffle(order)
    hold = [rows[i] for i in order[:a.hold]]
    train = [rows[i] for i in order[a.hold:]]
    json.dump([r["audio"] for r in hold], open(os.path.join(out, "отложены.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)

    t0 = time.time()
    m = Qwen3TTSModel.from_pretrained(a.init, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                      device_map=dev)
    model = m.model
    model.speech_tokenizer = None          # коды звука уже посчитаны - кодировщик не нужен
    if dev == "mps":
        torch.mps.empty_cache()
    for p in model.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGET),
                            model.talker.model)
    # пересчёт активаций: пик 6,6 -> 4,1 ГБ на фразе 15 с (замер 20.10), медленнее на ~25 %
    model.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    named = [(n, p) for n, p in model.talker.model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    say(f"{a.voice}: модель за {time.time() - t0:.0f} с, LoRA {sum(p.numel() for p in params) / 1e6:.1f} млн, "
        f"фраз {len(train)} + {len(hold)} отложено, эпох {a.epochs}, lr {a.lr}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    config = AutoConfig.from_pretrained(a.init)
    tc = config.talker_config
    T = model.talker
    helper = TTSDataset([], m.processor, config)          # только мел образца
    mels: dict[str, object] = {}

    def item(r):
        ids = m.processor(text=f"<|im_start|>assistant\n{r['text']}<|im_end|>\n<|im_start|>assistant\n",
                          return_tensors="pt")["input_ids"]
        if r["ref_audio"] not in mels:
            wav, sr = helper._normalize_audio_inputs(r["ref_audio"])[0]
            mels[r["ref_audio"]] = helper.extract_mels(wav, sr)
        return ids.reshape(1, -1), torch.tensor(r["audio_codes"], dtype=torch.long), mels[r["ref_audio"]]

    def emb_ids(ids):
        return T.get_input_embeddings()(torch.tensor([ids], device=dev))

    def loss_of(it):
        """Потеря в той же форме, в какой Herald потом озвучивает (mlx-audio, только вектор
        голоса, язык auto): [роль 3] [tts_pad ×4, tts_bos + nothink, think_bos, think_eos,
        голос, pad] [1-й знак текста + codec_bos] [кадр k + k-й знак текста (потом eos, pad)].
        Выход на позиции 8 + k - код 0 кадра k (последний - codec_eos); sub-talker берёт
        скрытое состояние той же позиции (оно видело кадры до k) и код 0 кадра k.
        Официальный sft_12hz.py (transformers 4.57) так не делает: текст без text_projection
        и метки сдвинуты дважды (потеря исходной модели 13,7 против 3,9 здесь) - 20.10 обучение
        по нему разучило модель читать."""
        ids, c, mel = it
        ids, c = ids.to(dev), c.to(dev)
        tE = T.text_projection(T.get_text_embeddings()(ids))
        bos, eos, pad = T.text_projection(T.get_text_embeddings()(torch.tensor(
            [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], device=dev))).split(1, 1)
        with torch.no_grad():
            spk = model.speaker_encoder(mel.to(dev, model.dtype)).reshape(1, 1, -1)
        cpre = torch.cat([emb_ids([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]), spk,
                          emb_ids([tc.codec_pad_id, tc.codec_bos_id])], 1)            # 6
        head = torch.cat([tE[:, :3], torch.cat([pad.expand(1, 4, -1), bos], 1) + cpre[:, :5],
                          tE[:, 3:4] + cpre[:, 5:6]], 1)                                # 9
        fr = T.get_input_embeddings()(c[:, 0])
        for j in range(1, 16):
            fr = fr + T.code_predictor.get_input_embeddings()[j - 1](c[:, j])
        n = c.shape[0]
        trail = torch.cat([tE[:, 4:-5], eos], 1)[:, :n]
        if trail.shape[1] < n:
            trail = torch.cat([trail, pad.expand(1, n - trail.shape[1], -1)], 1)
        x = torch.cat([head, fr[None] + trail], 1)                                     # 9 + n
        o = T(inputs_embeds=x, output_hidden_states=True)
        tgt = torch.cat([c[:, 0], torch.tensor([tc.codec_eos_token_id], device=dev)])
        lg = o.logits[0, 8:8 + n + 1].float()
        l0 = torch.nn.functional.cross_entropy(lg, tgt)
        hs = o.hidden_states[0][-1][0, 8:8 + n]
        sl, _ = T.forward_sub_talker_finetune(c, hs)
        l1 = torch.nn.functional.cross_entropy(sl.float().reshape(-1, sl.shape[-1]), c[:, 1:].reshape(-1))
        return l0 + 0.3 * l1

    it_tr, it_ho = [item(r) for r in train], [item(r) for r in hold]

    def batches(items, shuffle):
        idx = list(range(len(items)))
        if shuffle:
            rnd.shuffle(idx)
        for i in idx:
            yield items[i]

    def held_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in batches(it_ho, False):
                v = float(loss_of(b).item())
                if v == v:                          # nan - фраза не в счёт
                    tot += v
                    n += 1
        model.train()
        return tot / max(n, 1)

    if a.probe:
        say(f"проба: потеря исходной модели на отложенных {held_loss():.3f}")
        return 0
    say(f"эпоха 0: потеря на отложенных {held_loss():.3f}")
    model.train()
    for ep in range(1, a.epochs + 1):
        t0, run, k = time.time(), 0.0, 0
        skipped = 0
        for b in batches(it_tr, True):
            loss = loss_of(b)
            if not torch.isfinite(loss):            # одна такая фраза испортила бы все веса
                skipped += 1
                say(f"эпоха {ep}: потеря не число - фраза пропущена ({skipped})")
                continue
            (loss / a.accum).backward()
            run += float(loss.item())
            k += 1
            if k % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
            if dev == "mps" and k % 25 == 0:
                torch.mps.empty_cache()
            if k % 100 == 0:
                mem = f", видеоядро {torch.mps.driver_allocated_memory() / 1e9:.1f} ГБ" if dev == "mps" else ""
                say(f"эпоха {ep}, пакет {k}: потеря {run / k:.3f}, {time.time() - t0:.0f} с{mem}")
        if k % a.accum:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
        sd = {n.replace(".default", ""): p.detach().to("cpu", torch.float32).contiguous() for n, p in named}
        save_file(sd, os.path.join(out, f"epoch_{ep:02d}.safetensors"))
        hl = held_loss()
        say(f"эпоха {ep}: потеря обучения {run / k:.3f}, на отложенных {hl:.3f}, {time.time() - t0:.0f} с")
        json.dump({"r": 16, "alpha": 32, "target_modules": TARGET, "base": a.init, "applied_to": "talker.model",
                   "voice": a.voice, "epochs_done": ep, "lr": a.lr, "batch": a.accum,
                   "phrases": len(train), "held_out": len(hold), "data": a.data, "held_loss": round(hl, 4),
                   "personal_use_only": True},
                  open(os.path.join(out, "adapter.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if dev == "mps":
            torch.mps.empty_cache()
    say("готово")
    return 0


if __name__ == "__main__":
    sys.exit(main())
