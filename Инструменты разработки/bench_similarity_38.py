#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Похожесть голоса на образец (3.8, после оценки Андрея 19.10).

Как в Seed-TTS-eval (SIM): косинус между векторами голоса (speaker embedding)
синтеза и образца. Векторы - модель проверки диктора на WavLM:
microsoft/wavlm-base-plus-sv (MIT; в Seed-TTS-eval - WavLM-large + ECAPA-TDNN,
у нас - базовая WavLM с x-vector: числа сопоставимы между движками, но не с
таблицами статей).

Меряется:
  * SIM - похожесть каждой записи на образец голоса (тот, по которому клонировали);
  * постоянство - средняя похожесть записей одного движка и голоса между собой
    (держит ли голос от фразы к фразе: Андрей слышал у Qwen «каждая фраза другим
    голосом»);
  * калибровка: живой чтец - 10 кусков по 8 с из ДРУГИХ его записей (потолок
    «свой голос»), и чужой голос - образец другого чтеца.

    python "Инструменты разработки/bench_similarity_38.py" [--dir Архив/... --engines ...]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import glob
import json
import subprocess

import numpy as np

MODEL = "microsoft/wavlm-base-plus-sv"
VOICES = {"Вячеслав Герасимов": "Образцы голосов/Вячеслав Герасимов 3. Куприн - Олеся.m4b",
          "Игорь Ященко": "Образцы голосов/Каменистый Артем - Альфа-ноль 11, Убийца легенд/0003 - Убийца легенд.mp3"}
_m = None


def _model():
    global _m
    if _m is None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        fe = AutoFeatureExtractor.from_pretrained(MODEL)
        m = WavLMForXVector.from_pretrained(MODEL).eval()
        _m = (fe, m, torch)
    return _m


def load16(path: str, start: float = 0.0, dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", path]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


KIND = "wavlm"            # или "ecapa" - speechbrain/spkrec-ecapa-voxceleb (Apache-2.0)
_ecapa = None


def emb(a16: np.ndarray) -> np.ndarray:
    global _ecapa
    if KIND == "ecapa":
        import torch
        if _ecapa is None:
            import huggingface_hub as hh
            from huggingface_hub import snapshot_download
            _dl = hh.hf_hub_download

            def _compat(*x, use_auth_token=None, **k):          # старый параметр speechbrain
                return _dl(*x, **k)
            hh.hf_hub_download = _compat
            from speechbrain.inference.speaker import EncoderClassifier
            # speechbrain 1.0 зовёт hf_hub_download со старым параметром - берём папку сами
            local = snapshot_download("speechbrain/spkrec-ecapa-voxceleb")
            _ecapa = EncoderClassifier.from_hparams(
                source=local, savedir=_os.path.expanduser("~/.cache/text2audio/models/ecapa-voxceleb"))
        with torch.inference_mode():
            e = _ecapa.encode_batch(torch.from_numpy(a16)[None]).reshape(-1).numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    fe, m, torch = _model()
    x = fe(a16, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        e = m(**x).embeddings[0].numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def ref_path(voice: str) -> str:
    from core.engines.f5 import default_voices_dir
    return _os.path.join(default_voices_dir(), voice + ".wav")


def pair_mean(E: list[np.ndarray]) -> float:
    if len(E) < 2:
        return float("nan")
    M = np.stack(E)
    S = M @ M.T
    n = len(E)
    return float((S.sum() - n) / (n * (n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Архив/сравнение движков 3.8")
    ap.add_argument("--engines", default="qwen,qwen4,f5,chatterbox,voxcpm2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", choices=["wavlm", "ecapa"], default="wavlm")
    a = ap.parse_args()
    global KIND
    KIND = a.model
    a.out = a.out or _os.path.join(a.dir, f"похожесть {a.model}.json")
    refs = {v: emb(load16(ref_path(v))) for v in VOICES}
    res = {}
    print("| что | чтец | похожесть на образец: среднее (мин-макс) | постоянство (записи между собой) |")
    print("|---|---|---|---|")
    # калибровка: живой чтец, другие его записи
    for v, src in VOICES.items():
        E = [emb(load16(src, 120 + 40 * k, 8)) for k in range(10)]
        s = [float(e @ refs[v]) for e in E]
        other = [u for u in VOICES if u != v][0]
        x = [float(e @ refs[other]) for e in E]
        res[f"живой|{v}"] = {"sim": s, "pair": pair_mean(E), "cross": x}
        print(f"| живой чтец (другие записи) | {v} | {np.mean(s):.3f} ({min(s):.2f}-{max(s):.2f}) | "
              f"{pair_mean(E):.3f} |", flush=True)
        print(f"| для сравнения: живой {v} против образца {other} | - | {np.mean(x):.3f} | - |", flush=True)
    for name in a.engines.split(","):
        for v in VOICES:
            files = sorted(glob.glob(_os.path.join(a.dir, name, f"{v}__*.wav")))
            if not files:
                continue
            E = [emb(x) for x in (load16(f) for f in files) if len(x) > 8000]   # тишину не меряем
            s = [float(e @ refs[v]) for e in E]
            res[f"{name}|{v}"] = {"sim": s, "pair": pair_mean(E)}
            print(f"| {name} | {v} | {np.mean(s):.3f} ({min(s):.2f}-{max(s):.2f}) | "
                  f"{pair_mean(E):.3f} |", flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
