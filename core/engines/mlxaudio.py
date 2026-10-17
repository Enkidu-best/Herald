"""Клоны через mlx-audio (3.8, задание 10а): Qwen3-TTS ru-stress, Chatterbox
Multilingual V3, VoxCPM2 - в отдельном процессе (core/engines/mlx_audio_worker.py)
и своём окружении .venv-mlxaudio.

Qwen3-TTS 1.7B ru-stress-cf (siriusfreak, Apache-2.0) дообучена выполнять знак
ударения U+0301 после ударной гласной (96 % против 17 % у базовой). Ударения
ставит RUAccent как обычно («+» перед гласной: пайплайн, «Как будет прочитано»,
единое ударение по книге - без изменений); здесь «+» переводится в U+0301.

Голос - тот же образец, что у F5 (папка голосов: <имя>.wav и <имя>.txt).
Пока прототип: только Mac с Apple Silicon и окружение разработчика.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading

import numpy as np

from .base import TTSEngine, Voice

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PY = os.path.join(HERE, ".venv-mlxaudio", "bin", "python")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_audio_worker.py")

# модель -> откуда и как с ней говорить
MODELS = {
    "qwen-stress": {"title": "Qwen3-TTS ru-stress", "repo": "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf",
                    "local": "qwen3-tts-ru-stress-{bits}bit", "lang": "Russian", "stress": "acute",
                    "license": "Apache-2.0"},
    "chatterbox": {"title": "Chatterbox Multilingual V3", "repo": "mlx-community/chatterbox-multilingual-v3",
                   "lang": "ru", "stress": None, "license": "MIT"},
    "voxcpm2": {"title": "VoxCPM2", "repo": "mlx-community/VoxCPM2-8bit", "lang": "Russian",
                "stress": None, "license": "Apache-2.0"},
}


def models_dir() -> str:
    from ..osutil import data_dir
    d = os.path.join(str(data_dir()), "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_path(model: str, bits: int = 8) -> str:
    """Где лежит модель: сконвертированная в MLX (Qwen ru-stress) или repo HF."""
    m = MODELS[model]
    if "local" in m:
        return os.path.join(models_dir(), m["local"].format(bits=bits))
    return m["repo"]


def available() -> bool:
    return os.path.exists(VENV_PY)


def plus_to_acute(text: str) -> str:
    """«з+амок» -> «за́мок»: U+0301 после ударной гласной (так учили ru-stress)."""
    from ..stress import to_combining
    return to_combining(text)


class MlxAudioEngine(TTSEngine):
    """Клон через модель mlx-audio. Говорит с отдельным процессом."""
    name = "mlxaudio"
    supports_cloning = True
    stress_format = "+"              # пайплайн ставит «+»; Qwen получает U+0301
    max_chunk_chars = 300

    def __init__(self, model: str = "qwen-stress", bits: int = 8, language: str = "ru",
                 seed: int | None = None, **_kw) -> None:
        super().__init__()
        self.model = model
        self.bits = bits
        self.language = language
        self.seed = seed
        self.sample_rate = 24000
        self._proc = None
        self._lock = threading.Lock()
        self.last_stats: dict = {}
        if MODELS[model]["stress"] is None:
            self.stress_format = None     # «+» модель прочитала бы вслух

    # --- голоса: те же образцы, что у F5 ---
    def list_voices(self) -> list[Voice]:
        from .f5 import F5Engine
        return F5Engine().list_voices()

    def cache_key(self) -> str:
        return f"mlxaudio|{self.model}|{model_path(self.model, self.bits)}|{self.bits}"

    # --- процесс ---
    def _call(self, d: dict) -> dict:
        with self._lock:
            self._proc.stdin.write(json.dumps(d, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("процесс mlx-audio завершился")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(f"mlx-audio: {r.get('error')}")
        return r

    def load(self) -> None:
        if not available():
            raise RuntimeError("нет окружения .venv-mlxaudio (см. Инструменты разработки/"
                               "setup_mlxaudio.sh)")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._proc = subprocess.Popen([VENV_PY, WORKER], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, encoding="utf-8", env=env)
        r = self._call({"op": "load", "model": model_path(self.model, self.bits)})
        self.sample_rate = int(r["sr"])
        self.load_stats = r

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.stdin.write('{"op": "quit"}\n')
                p.stdin.flush()
                p.wait(timeout=10)
            except Exception:
                p.kill()
        self._loaded = False

    def _ref(self, voice: str) -> tuple[str, str]:
        from .f5 import default_voices_dir
        d = default_voices_dir()
        wav = os.path.join(d, voice + ".wav")
        txt = os.path.join(d, voice + ".txt")
        ref_text = open(txt, encoding="utf-8").read().replace("+", "").strip() \
            if os.path.exists(txt) else None
        return wav, ref_text

    def synth_chunk(self, text: str, voice: str, speed_mul: float = 1.0, seed=None,
                    stream: bool = False, **kw) -> np.ndarray:
        import soundfile as sf
        self.ensure_loaded()
        if MODELS[self.model]["stress"] == "acute":
            text = plus_to_acute(text)
        else:
            text = text.replace("+", "")
        wav, ref_text = self._ref(voice)
        out = tempfile.mktemp(suffix=".wav", prefix="herald_mlxa_")
        try:
            r = self._call({"op": "synth", "text": text, "out": out, "ref_audio": wav,
                            "ref_text": ref_text, "lang": MODELS[self.model]["lang"],
                            "seed": seed if seed is not None else self.seed, "stream": stream})
            self.last_stats = r
            a, _sr = sf.read(out, dtype="float32")
        finally:
            try:
                os.remove(out)
            except OSError:
                pass
        # темп: у этих моделей параметра нет - пайплайн растягивает звук после
        # синтеза (atempo), как у готовых голосов; speed_mul куска не применяем
        return a
