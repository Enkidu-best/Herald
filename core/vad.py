"""Детектор речи Silero VAD (ONNX, ~2 МБ, в assets/models, без интернета).

Тот же, что в Scribe: github.com/snakers4/silero-vad (MIT). Здесь он нужен,
чтобы найти, где в куске синтеза кончается РЕЧЬ, а не громкость: тихое
безударное окончание («правил[о]») падает ниже порога громкости и раньше
отрезалось вместе с тишиной.
"""

from __future__ import annotations

import os

import numpy as np

SR = 16000
CHUNK = 512                     # 32 мс
CTX = 64
HOP = CHUNK / SR
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "models", "silero_vad.onnx")
THRESHOLD = 0.5                 # начало речи
NEG_THRESHOLD = 0.35            # конец речи (гистерезис)

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        _session = ort.InferenceSession(MODEL_PATH, sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    return _session


def _to16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return audio.astype(np.float32)
    n = int(len(audio) * SR / sr)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def speech_probs(audio: np.ndarray, sr: int) -> np.ndarray:
    """Вероятность речи для каждого кадра 32 мс."""
    a = _to16k(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
    n = len(a) // CHUNK
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    sess = _get_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros((1, CTX), dtype=np.float32)
    srv = np.array(SR, dtype=np.int64)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ch = a[i * CHUNK:(i + 1) * CHUNK][None, :]
        p, state = sess.run(None, {"input": np.concatenate([ctx, ch], axis=1),
                                   "state": state, "sr": srv})
        ctx = ch[:, -CTX:]
        out[i] = p[0, 0]
    return out


def speech_end(audio: np.ndarray, sr: int) -> float | None:
    """Секунда, где кончается речь (по гистерезису), или None, если речи нет."""
    p = speech_probs(audio, sr)
    on = np.where(p >= THRESHOLD)[0]
    if not len(on):
        return None
    i = int(on[-1])
    while i + 1 < len(p) and p[i + 1] >= NEG_THRESHOLD:
        i += 1
    return (i + 1) * HOP
