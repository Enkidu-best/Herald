"""Распознавание речи для проверок и для расшифровки образца - на любой системе.

Apple Silicon - mlx-whisper (видеоядро Apple); Windows и Linux - faster-whisper
(видеокарта NVIDIA, если есть, иначе процессор), как в Scribe 2.4. Результат в
одном виде: {"text", "language", "segments": [{"start", "end", "text",
"avg_logprob", "compression_ratio", "words": [{"word", "start", "end"}]}]}.

Модели по назначению:
  small - дешёвая проверка краёв куска;
  turbo - строгая проверка всего куска и расшифровка образца (large-v3-turbo);
  large - судья в стендах (large-v3, не turbo - чтобы не мерить проверку ею же).
"""

from __future__ import annotations

import contextlib
import io
import os

MLX = {"small": "mlx-community/whisper-small-mlx",
       "turbo": "mlx-community/whisper-large-v3-turbo",
       "large": "mlx-community/whisper-large-v3-mlx"}
FW = {"small": "small", "turbo": "large-v3-turbo", "large": "large-v3"}
_fw_models: dict = {}
_backend: str | None = None


def backend() -> str:
    """mlx или fw (faster-whisper)."""
    global _backend
    if _backend is None:
        _backend = "fw"
        want = os.environ.get("HERALD_ASR")
        if want in ("mlx", "fw"):
            _backend = want
        else:
            from .engines import apple_silicon
            if apple_silicon():
                try:
                    import mlx_whisper  # noqa: F401
                    _backend = "mlx"
                except Exception:
                    pass              # нет MLX - faster-whisper
    return _backend


def _fw(kind: str):
    if kind not in _fw_models:
        from faster_whisper import WhisperModel
        dev = "cpu"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                dev = "cuda"
        except Exception:
            pass                      # без CUDA - процессор
        name = FW.get(kind, kind)
        try:
            _fw_models[kind] = WhisperModel(name, device=dev,
                                            compute_type="float16" if dev == "cuda" else "int8")
        except Exception:
            if dev != "cuda":
                raise
            _fw_models[kind] = WhisperModel(name, device="cpu", compute_type="int8")
    return _fw_models[kind]


def _load16k(path: str):
    """Звук для faster-whisper - через ffmpeg, а не через PyAV: свежая PyAV
    несовместима с faster-whisper 1.2 (`metadata_errors`), и на чистой установке
    распознавание не работало вовсе (поймано на раннере GitHub)."""
    import shutil
    import subprocess
    import numpy as np
    exe = shutil.which("ffmpeg") or "ffmpeg"
    raw = subprocess.run([exe, "-nostdin", "-v", "error", "-i", path, "-f", "f32le",
                          "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def transcribe(path: str, kind: str = "turbo", language: str | None = None,
               word_timestamps: bool = False) -> dict:
    if backend() == "mlx":
        from .mlxthread import run      # MLX - только в одной нити процесса (задание 14 В)

        def go():
            import mlx_whisper
            with contextlib.redirect_stdout(io.StringIO()):
                return mlx_whisper.transcribe(path, path_or_hf_repo=MLX.get(kind, kind),
                                              language=language, verbose=False,
                                              word_timestamps=word_timestamps,
                                              condition_on_previous_text=False)
        return run(go)
    segs, info = _fw(kind).transcribe(_load16k(path), language=language,
                                      word_timestamps=word_timestamps,
                                      condition_on_previous_text=False)
    out = []
    for s in segs:
        d = {"start": s.start, "end": s.end, "text": s.text,
             "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio}
        if word_timestamps:
            d["words"] = [{"word": w.word, "start": w.start, "end": w.end}
                          for w in (s.words or [])]
        out.append(d)
    return {"text": " ".join(d["text"].strip() for d in out),
            "language": getattr(info, "language", language) or language,
            "segments": out}
