#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, шаг 2: данные для дообучения Qwen3-TTS на голосе чтеца.

Из аудиокниг чтеца (Образцы голосов/) - фразы 3-15 с с расшифровкой:
  * Whisper large-v3-turbo по длинному файлу целиком - сегменты с пунктуацией и
    временами; сегменты склеиваются в фразы 3-15 с по границам предложений;
  * чистка: уверенность распознавания (avg_logprob > -0,6), повторы
    (compression_ratio < 2,2), перегруз (сэмплов у потолка < 0,1 %), без музыки и
    пустот (доля речи по VAD > 0,6);
  * текст - с ударениями RUAccent, как в конвейере Herald, и знаком U+0301 после
    ударной гласной (как учили ru-stress);
  * формат - как у официального дообучения Qwen3-TTS (QwenLM/Qwen3-TTS,
    finetuning/): JSONL {"audio", "text", "ref_audio"}, один ref_audio на всё.
Пишет ~/.cache/text2audio/train/<чтец>/ (wavs/, train_raw.jsonl, сводка.json).
Записи чтецов - только для личного использования: в git и в открытый доступ не идут.

    python "Инструменты разработки/prepare_voice_data_39.py" [--hours 2.0] [--voice ...]
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

SRC = {"Вячеслав Герасимов": ["Образцы голосов/Вячеслав Герасимов 3. Куприн - Олеся.m4b",
                              "Образцы голосов/Вячеслав Герасимов 2.mp3"],
       "Игорь Ященко": sorted(glob.glob("Образцы голосов/Каменистый Артем - Альфа-ноль 11, "
                                        "Убийца легенд/00*.mp3"))}
OUT = _os.path.expanduser("~/.cache/text2audio/train")
SR = 24000


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def utterances(segs: list[dict], lo: float = 3.0, hi: float = 15.0) -> list[dict]:
    """Сегменты Whisper -> фразы lo-hi с: копим сегменты до конца предложения."""
    out, cur = [], None
    for s in segs:
        t = s["text"].strip()
        if not t:
            continue
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": t, "lp": [s["avg_logprob"]],
                   "cr": [s["compression_ratio"]], "ns": [s.get("no_speech_prob", 0.0)]}
        elif s["end"] - cur["start"] <= hi:
            cur["end"], cur["text"] = s["end"], cur["text"] + " " + t
            cur["lp"].append(s["avg_logprob"])
            cur["cr"].append(s["compression_ratio"])
            cur["ns"].append(s.get("no_speech_prob", 0.0))
        else:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": t, "lp": [s["avg_logprob"]],
                   "cr": [s["compression_ratio"]], "ns": [s.get("no_speech_prob", 0.0)]}
        if cur and cur["text"][-1:] in ".!?…" and cur["end"] - cur["start"] >= lo:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return [u for u in out if lo <= u["end"] - u["start"] <= hi]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--voice", default=None)
    a = ap.parse_args()
    import soundfile as sf
    from core import asr, vad
    from core.stress import add_stress_text, to_combining
    from core.normalize import normalize_text
    from core.engines.f5 import default_voices_dir
    for voice, files in SRC.items():
        if a.voice and voice != a.voice:
            continue
        d = _os.path.join(OUT, voice)
        _os.makedirs(_os.path.join(d, "wavs"), exist_ok=True)
        rows, total, drop = [], 0.0, {"распознавание": 0, "повторы": 0, "перегруз": 0, "мало речи": 0}
        n = 0
        for f in files:
            if total >= a.hours * 3600:
                break
            print(f"{voice}: {_os.path.basename(f)}", flush=True)
            audio = decode(f)
            tmp = _os.path.join(d, "_whole.wav")
            sf.write(tmp, audio, SR)
            res = asr.transcribe(tmp, kind="turbo", language="ru")
            _os.remove(tmp)
            for u in utterances(res.get("segments", [])):
                if total >= a.hours * 3600:
                    break
                if min(u["lp"]) < -0.6 or max(u["ns"]) > 0.4:
                    drop["распознавание"] += 1
                    continue
                if max(u["cr"]) > 2.2:
                    drop["повторы"] += 1
                    continue
                s0, s1 = int(max(0, u["start"] - 0.05) * SR), int((u["end"] + 0.15) * SR)
                x = audio[s0:s1]
                if np.mean(np.abs(x) > 0.985) > 0.001:
                    drop["перегруз"] += 1
                    continue
                p = vad.speech_probs(x, SR)
                if len(p) and float(np.mean(p > 0.5)) < 0.6:
                    drop["мало речи"] += 1
                    continue
                n += 1
                wav = _os.path.join(d, "wavs", f"{n:05d}.wav")
                sf.write(wav, x, SR)
                plain = normalize_text(u["text"], expand_numbers=True, language="ru")
                marked = to_combining(add_stress_text(plain))
                rows.append({"audio": wav, "text": marked, "text_plain": plain,
                             "ref_audio": _os.path.join(default_voices_dir(), voice + ".wav"),
                             "dur": round(len(x) / SR, 2)})
                total += len(x) / SR
        with open(_os.path.join(d, "train_raw.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        info = {"чтец": voice, "фраз": len(rows), "часов": round(total / 3600, 2),
                "средняя длина, с": round(total / max(1, len(rows)), 1), "отброшено": drop}
        json.dump(info, open(_os.path.join(d, "сводка.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(info, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
