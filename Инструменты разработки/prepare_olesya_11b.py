#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б (решение Андрея 21.10 16:05): данные Герасимова - только «Олеся» (Куприн, 3 ч 10 мин),
нарезка ЗАНОВО - по паузам между словами, фразы 3-10 с.

Прежняя нарезка (prepare_voice_data_39.py) склеивала сегменты Whisper до конца предложения в
3-15 с: длинные предложения Куприна туда не помещались (взято 1,25 ч из 3 ч), метки сегментов
неточны (край фразы резал слово), а F5 на Mac учится только на фразах до ~9,6 с (длиннее -
не помещается в память). Здесь:
  * Whisper large-v3-turbo по всей книге со словами и их временем (пунктуация - у слов);
  * фраза режется на паузе между словами (≥ 0,3 с, или ≥ 0,15 с после знака препинания),
    длина 3-10 с; граница - посередине тишины, а не по метке слова;
  * отбор как раньше: уверенность распознавания, повторы, перегруз, доля речи (VAD);
  * текст - слова Whisper с пунктуацией, нормализация и ударения - как в конвейере.
Дальше - та же чистка, что у Ященко (clean_train_11a.py: повторное распознавание фразы, CER ≤ 5 %,
края). Пишет <чтец>/olesya/wavs/NNNNN.wav и train_olesya_raw.jsonl. Под ../.heavy-lock.

    python "Инструменты разработки/prepare_olesya_11b.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import re
import subprocess
import time

import numpy as np

SRC = "Образцы голосов/Вячеслав Герасимов 3. Куприн - Олеся.m4b"
VOICE = "Вячеслав Герасимов"
SR = 24000
LO, HI = 3.0, 10.0
GAP, GAP_PUNCT = 0.30, 0.15


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def _good(words, i) -> bool:
    """После слова i - хорошая пауза для разреза."""
    if i + 1 >= len(words):
        return True
    gap = words[i + 1]["start"] - words[i]["end"]
    return gap >= GAP or (bool(re.search(r"[.,!?;:…—–-]$", words[i]["word"].strip())) and gap >= GAP_PUNCT)


def phrases(words: list[dict]) -> list[list[dict]]:
    """Слова -> фразы 3-10 с: режем на хорошей паузе, когда фраза уже от 7 с или следующей
    хорошей паузы до 10 с не будет; нет хорошей паузы до 10 с - фразу не берём."""
    out, s = [], 0
    i = 0
    while i < len(words):
        dur = words[i]["end"] - words[s]["start"]
        if dur > HI:
            # от s до 10 с хорошей паузы не нашлось - начать со следующей паузы
            k = next((j for j in range(i, len(words)) if _good(words, j)), len(words) - 1)
            s = i = k + 1
            continue
        if _good(words, i) and dur >= LO:
            later = any(_good(words, j) and words[j]["end"] - words[s]["start"] <= HI
                        for j in range(i + 1, len(words)) if words[j]["end"] - words[s]["start"] <= HI)
            if dur >= 7.0 or not later:
                out.append(words[s:i + 1])
                s = i + 1
        i += 1
    return out


def main() -> int:
    import soundfile as sf
    from core import asr, vad
    from core.stress import add_stress_text, to_combining
    from core.normalize import normalize_text
    from core.engines.f5 import default_voices_dir
    d = _os.path.expanduser(f"~/.cache/text2audio/train/{VOICE}")
    wd = _os.path.join(d, "olesya", "wavs")
    _os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    audio = decode(SRC)
    tmp = _os.path.join(d, "_olesya.wav")
    sf.write(tmp, audio, SR)
    res = asr.transcribe(tmp, kind="turbo", language="ru", word_timestamps=True)
    _os.remove(tmp)
    words = []
    for s in res.get("segments", []):
        for w in s.get("words") or []:
            if w["word"].strip():
                words.append({**w, "lp": s.get("avg_logprob", 0.0), "cr": s.get("compression_ratio", 1.0),
                              "ns": s.get("no_speech_prob", 0.0)})
    print(f"распознано слов: {len(words)} за {time.time() - t0:.0f} с", flush=True)
    rows, drop, n = [], {"распознавание": 0, "повторы": 0, "перегруз": 0, "мало речи": 0}, 0
    ph = phrases(words)
    for k, p in enumerate(ph):
        if min(w["lp"] for w in p) < -0.6 or max(w["ns"] for w in p) > 0.4:
            drop["распознавание"] += 1
            continue
        if max(w["cr"] for w in p) > 2.2:
            drop["повторы"] += 1
            continue
        # граница - посередине тишины до / после фразы (не по метке слова)
        prev_end = ph[k - 1][-1]["end"] if k else max(0.0, p[0]["start"] - 0.3)
        next_start = ph[k + 1][0]["start"] if k + 1 < len(ph) else p[-1]["end"] + 0.3
        a0 = max(prev_end, p[0]["start"] - min(0.15, (p[0]["start"] - prev_end) / 2))
        a1 = min(next_start, p[-1]["end"] + min(0.2, (next_start - p[-1]["end"]) / 2))
        x = audio[int(a0 * SR):int(a1 * SR)]
        if np.mean(np.abs(x) > 0.985) > 0.001:
            drop["перегруз"] += 1
            continue
        pr = vad.speech_probs(x, SR)
        if len(pr) and float(np.mean(pr > 0.5)) < 0.6:
            drop["мало речи"] += 1
            continue
        n += 1
        wav = _os.path.join(wd, f"{n:05d}.wav")
        sf.write(wav, x, SR)
        text = " ".join(w["word"].strip() for w in p)
        plain = normalize_text(text, expand_numbers=True, language="ru")
        rows.append({"audio": wav, "text": to_combining(add_stress_text(plain)), "text_plain": plain,
                     "ref_audio": _os.path.join(default_voices_dir(), VOICE + ".wav"),
                     "dur": round(len(x) / SR, 2)})
    with open(_os.path.join(d, "train_olesya_raw.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"источник": SRC, "фраз": len(rows), "часов": round(sum(r["dur"] for r in rows) / 3600, 2),
            "средняя, с": round(float(np.mean([r["dur"] for r in rows])), 1) if rows else 0, "отброшено": drop}
    json.dump(info, open(_os.path.join(d, "сводка олеся.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
