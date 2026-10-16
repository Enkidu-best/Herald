#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отсев быстрых голосов (3.6, п. 2): Piper, Vosk TTS и нынешние Silero.

На каждый голос - один и тот же абзац (русский или английский):
  * первый звук - счёт первой фразы с уже загруженной моделью (цель ≤ 1 с);
  * RTF - счёт абзаца на секунду звука;
  * разборчивость - доля слов абзаца, которые Whisper large-v3-turbo слышит;
  * брак - перегруз (сэмплы у потолка) и тишина вместо речи;
  * высота голоса (медиана F0) - муж. / жен., если пол не подписан.
Звук каждого - в --out-dir (для слепого набора), числа - в JSON.

    python "Инструменты разработки/bench_fast_voices.py" --out-dir голоса/ --out голоса.json
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import time

import numpy as np

RU_TEXT = ("Ветер с залива к вечеру переменился и потянул с моря, принося запах соли и "
           "мокрого камня. Смотритель маяка закрыл ставни, проверил фитиль и сел к столу, "
           "где остывал чай в жестяной кружке. За окном темнело быстро, как всегда в октябре.")
EN_TEXT = ("By evening the wind had shifted and come in from the sea, bringing the smell of "
           "salt and wet stone. The lighthouse keeper closed the shutters, checked the wick and "
           "sat down at the table, where his tea was going cold in a tin mug.")
FIRST = {"ru": "Ветер с залива к вечеру переменился и потянул с моря.",
         "en": "By evening the wind had shifted and come in from the sea."}

PIPER = [  # (голос, пол по карточке, язык)
    ("ru_RU-denis-medium", "м", "ru"), ("ru_RU-dmitri-medium", "м", "ru"),
    ("ru_RU-irina-medium", "ж", "ru"), ("ru_RU-ruslan-medium", "м", "ru"),
    ("en_US-amy-medium", "ж", "en"), ("en_US-lessac-medium", "ж", "en"),
    ("en_US-kristin-medium", "ж", "en"), ("en_US-hfc_female-medium", "ж", "en"),
    ("en_US-ryan-medium", "м", "en"), ("en_US-joe-medium", "м", "en"),
    ("en_US-hfc_male-medium", "м", "en"), ("en_GB-alan-medium", "м", "en"),
    ("en_GB-cori-medium", "ж", "en"),
]
VOSK_MODEL = "vosk-model-tts-ru-0.9-multi"


def piper_paths(name: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download
    lang, who, q = name.split("-")
    base = f"{lang.split('_')[0]}/{lang}/{who}/{q}/{name}"
    return (hf_hub_download("rhasspy/piper-voices", base + ".onnx"),
            hf_hub_download("rhasspy/piper-voices", base + ".onnx.json"))


def f0_median(a: np.ndarray, sr: int) -> float:
    """Медиана основного тона (автокорреляция по кадрам 40 мс) - для муж. / жен."""
    w = int(0.04 * sr)
    vals = []
    for i in range(0, len(a) - w, w):
        x = a[i:i + w] - a[i:i + w].mean()
        if np.sqrt(np.mean(x ** 2)) < 0.02:
            continue
        c = np.correlate(x, x, "full")[w - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            vals.append(sr / k)
    return float(np.median(vals)) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="piper,vosk,silero")
    a = ap.parse_args()
    only = set(a.only.split(","))
    # python.org Python на Mac не видит системные сертификаты - Vosk не скачал бы
    # даже список моделей («model does not exist»)
    try:
        import certifi
        _os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    import soundfile as sf
    import difflib
    import re
    from core import asr
    import core.pipeline as P
    _os.makedirs(a.out_dir, exist_ok=True)

    def words(s):
        return re.sub(r"[^\w ]", " ", s.lower().replace("ё", "е").replace("+", "")).split()

    def judge(audio, sr, lang, text):
        path = _os.path.join(a.out_dir, "_asr.wav")
        sf.write(path, audio, sr)
        heard = words(asr.transcribe(path, "turbo", language=lang)["text"])
        want = words(text)
        m = difflib.SequenceMatcher(None, want, heard, autojunk=False)
        return sum(b.size for b in m.get_matching_blocks()) / max(1, len(want))

    rows = []

    def record(engine, vid, gender, lang, synth, sr, load_sec):
        text = RU_TEXT if lang == "ru" else EN_TEXT
        synth(FIRST[lang])                                 # прогрев
        t0 = time.monotonic()
        synth(FIRST[lang])
        first = time.monotonic() - t0
        t0 = time.monotonic()
        audio = np.asarray(synth(text), dtype=np.float32).reshape(-1)
        took = time.monotonic() - t0
        dur = len(audio) / sr
        f0 = f0_median(audio, sr)
        row = {"движок": engine, "голос": vid, "пол_карточка": gender, "язык": lang,
               "загрузка_с": round(load_sec, 2), "первый_звук_с": round(first, 2),
               "RTF": round(took / max(dur, 0.1), 3), "звук_с": round(dur, 1),
               "разборчивость": round(judge(audio, sr, lang, text), 3),
               "у_потолка_%": round(100 * float(np.mean(np.abs(audio) > 0.99)), 3),
               "F0_Гц": round(f0, 0), "пол_по_голосу": "ж" if f0 >= 165 else "м"}
        path = _os.path.join(a.out_dir, f"{engine}__{vid}.wav")
        sf.write(path, audio, sr)
        row["файл"] = path
        rows.append(row)
        print("   ", row, flush=True)

    print("Piper:", flush=True)
    from piper import PiperVoice
    for vid, g, lang in (PIPER if "piper" in only else []):
        try:
            t0 = time.monotonic()
            onnx, cfg = piper_paths(vid)
            v = PiperVoice.load(onnx, config_path=cfg)
            load = time.monotonic() - t0

            def synth(t, v=v):
                return np.concatenate([c.audio_float_array for c in v.synthesize(t)])
            record("piper", vid, g, lang, synth, v.config.sample_rate, load)
        except Exception as e:
            print(f"    {vid}: не вышло - {e}", flush=True)

    print("Vosk TTS:", flush=True)
    try:
        if "vosk" not in only:
            raise LookupError("пропущен")
        from vosk_tts import Model, Synth
        t0 = time.monotonic()
        model = Model(model_name=VOSK_MODEL)
        syn = Synth(model)
        load = time.monotonic() - t0
        cfg = getattr(model, "config", None) or {}
        n_spk = int(cfg.get("num_speakers") or cfg.get("model", {}).get("num_speakers") or 5)
        vsr = int(cfg.get("audio", {}).get("sample_rate") or 22050)
        print(f"    дикторов {n_spk}, {vsr} Гц", flush=True)
        for spk in range(n_spk):
            for stressed in (False, True):
                def synth(t, spk=spk, stressed=stressed):
                    if stressed:
                        t = P._prepare_text(t, "+", True, True, "ru")
                    out = np.asarray(syn.synth_audio(t, speaker_id=spk))
                    return (out.astype(np.float32) / 32768.0 if out.dtype == np.int16
                            else out.astype(np.float32))
                record("vosk", f"{VOSK_MODEL}#{spk}{'+' if stressed else ''}", "", "ru",
                       synth, vsr, load)
    except Exception as e:
        print(f"    Vosk: не вышло - {e}", flush=True)

    print("Silero (нынешние):", flush=True)
    if "silero" not in only:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return 0
    from core.engines import get_engine
    from core.engines.silero import VOICES
    eng = get_engine("silero")
    eng.ensure_loaded()
    for v in VOICES:
        def synth(t, v=v):
            return eng.synth_chunk(P._prepare_text(t, "+", True, True, "ru"), v.id)
        record("silero", v.id, v.gender, "ru", synth, eng.sample_rate, 0.0)

    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
