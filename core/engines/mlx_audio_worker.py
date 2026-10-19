#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отдельный процесс для моделей mlx-audio (3.8, задание 10а): Qwen3-TTS
ru-stress, Chatterbox Multilingual V3, VoxCPM2.

Почему отдельно: mlx-audio 0.5.9 требует transformers >= 5.14 и свежий mlx, а
F5 и выравниватель Herald живут на своих версиях. Поэтому у этих моделей своё
окружение (.venv-mlxaudio) и свой процесс; Herald (и сервер /speak) их не
трогает. Herald ничего не импортирует отсюда - только запускает.

Протокол: по строке JSON на вход (stdin), по строке JSON на выход (stdout).
    {"op": "load", "model": "<папка или repo>"}
        -> {"ok": true, "sr": 24000, "sec": <загрузка, с>, "mem_gb": ...}
    {"op": "synth", "text": "...", "out": "<wav>", "ref_audio": "<wav>",
     "ref_text": "...", "lang": "Russian", "seed": 1, "stream": false}
        -> {"ok": true, "audio_sec": ..., "wall": ..., "ttfb": ..., "peak_gb": ...}
    {"op": "quit"}
Ошибка -> {"ok": false, "error": "..."}; процесс живёт дальше.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
import traceback

_model = None
_sr = 24000
# ответы - в настоящий stdout; всё, что печатают сами библиотеки, - в stderr
# (иначе их строки ломают протокол)
_OUT = sys.stdout
sys.stdout = sys.stderr


def _reply(d: dict) -> None:
    _OUT.write(json.dumps(d, ensure_ascii=False) + "\n")
    _OUT.flush()


def _load(model: str) -> dict:
    global _model, _sr
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    t0 = time.monotonic()
    _model = load_model(model)
    _sr = int(getattr(_model, "sample_rate", None) or getattr(_model, "sr", None) or 24000)
    return {"sr": _sr, "sec": round(time.monotonic() - t0, 2),
            "mem_gb": round(mx.get_active_memory() / 1e9, 2)}


_orig_spk = None


def _centroid(d: dict) -> dict:
    """Средний вектор голоса по фрагментам (папка wav 24 кГц) вместо вектора образца
    (3.9: голос ровнее). {"dir": None} - вернуть обычный."""
    global _orig_spk
    import glob as _g
    import mlx.core as mx
    from mlx_audio.utils import load_audio
    if _orig_spk is None:
        _orig_spk = _model.extract_speaker_embedding
    if not d.get("dir"):
        _model.extract_speaker_embedding = _orig_spk
        return {"n": 0}
    embs = []
    for f in sorted(_g.glob(d["dir"] + "/*.wav")):
        a = load_audio(f, sample_rate=24000)
        embs.append(_orig_spk(a[None] if a.ndim == 1 else a))
    c = mx.mean(mx.concatenate(embs, axis=0), axis=0, keepdims=True)
    mx.eval(c)
    _model.extract_speaker_embedding = lambda audio, sr=24000, _c=c: _c
    return {"n": len(embs)}


def _kwargs(fn, d: dict) -> dict:
    """Параметры, которые эта модель понимает (у моделей mlx-audio они разные)."""
    try:
        names = set(inspect.signature(fn).parameters)
        var = any(p.kind == p.VAR_KEYWORD for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        names, var = set(), True
    want = {"ref_audio": d.get("ref_audio"), "ref_text": d.get("ref_text")}
    lang = d.get("lang")
    if lang:
        for k in ("language", "lang_code", "lang"):
            if k in names:
                want[k] = lang
                break
    for k, v in (d.get("extra") or {}).items():
        want[k] = v
    return {k: v for k, v in want.items() if v is not None and (var or k in names)}


def _synth(d: dict) -> dict:
    import numpy as np
    import mlx.core as mx
    import soundfile as sf
    if _model is None:
        raise RuntimeError("модель не загружена")
    if d.get("seed") is not None:
        mx.random.seed(int(d["seed"]))
    mx.reset_peak_memory()
    kw = _kwargs(_model.generate, d)
    stream = bool(d.get("stream"))
    if stream:
        kw["stream"] = True
    t0 = time.monotonic()
    ttfb = None
    parts = []
    for r in _model.generate(text=d["text"], **kw):
        if ttfb is None:
            ttfb = time.monotonic() - t0
        parts.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
    wall = time.monotonic() - t0
    a = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    sf.write(d["out"], a, _sr)
    return {"audio_sec": round(len(a) / _sr, 3), "wall": round(wall, 3),
            "ttfb": round(ttfb or wall, 3), "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "sr": _sr}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            op = d.get("op")
            if op == "quit":
                _reply({"ok": True})
                return 0
            if op == "load":
                _reply({"ok": True, **_load(d["model"])})
            elif op == "centroid":
                _reply({"ok": True, **_centroid(d)})
            elif op == "synth":
                _reply({"ok": True, **_synth(d)})
            else:
                _reply({"ok": False, "error": f"неизвестная команда {op}"})
        except Exception as e:
            _reply({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()[-1500:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
