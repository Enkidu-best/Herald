#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Путь 2 (3.8, после оценки Андрея 19.10): Qwen ru-stress (ударения) -> kNN-VC
(тембр нашего чтеца).

kNN-VC - Baas, van Niekerk, Kamper, «Voice Conversion With Just Nearest
Neighbors», Interspeech 2023 (arXiv 2305.18975), github.com/bshall/knn-vc, MIT.
Признаки WavLM-Large каждого кадра исходной речи заменяются средним k ближайших
кадров из записей целевого чтеца (5-10 мин), затем вокодер HiFi-GAN (16 кГц).
Содержание и интонация (а значит и ударения) - от источника, тембр - от чтеца.

    python "Инструменты разработки/knnvc_probe.py" rs_xvec [lora05 ...]
Пишет Архив/qwen голос 3.8/<режим>_knnvc/<чтец>__NN.wav. Под ../.heavy-lock.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import time

OUT = "Архив/qwen голос 3.8"
VOICES = ("Вячеслав Герасимов", "Игорь Ященко")


def main() -> int:
    import torch
    import torchaudio
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                            pretrained=True, device="cpu")
    for src in _sys.argv[1:] or ["rs_xvec"]:
        dst = _os.path.join(OUT, src + "_knnvc")
        _os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        total = 0.0
        for v in VOICES:
            # записи чтеца - кусками по 20 с: целиком (10 мин) WavLM-Large не помещается в память
            target = knn_vc.get_matching_set(sorted(glob.glob(_os.path.join(OUT, "цель", v, "*.wav"))))
            for f in sorted(glob.glob(_os.path.join(OUT, src, f"{v}__*.wav"))):
                wav, sr = torchaudio.load(f)
                if wav.shape[-1] < sr // 2:
                    continue                   # тишина - нечего преобразовывать
                tmp = f + ".16k.wav"
                torchaudio.save(tmp, torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000),
                                16000)
                q = knn_vc.get_features(tmp)
                out = knn_vc.match(q, target, topk=4)
                torchaudio.save(_os.path.join(dst, _os.path.basename(f)), out[None], 16000)
                total += out.shape[-1] / 16000
                _os.remove(tmp)
        print(f"{src}: преобразовано {total:.0f} с звука за {time.time() - t0:.0f} с", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
