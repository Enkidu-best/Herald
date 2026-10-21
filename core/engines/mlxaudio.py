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


def own_voices_dir() -> str:
    """Свои голоса ru-stress («Диктор 1-3», 11а): образцы из её же записей - их
    видит только этот движок (F5 такой образец не нужен). Голоса для личного
    использования: в git и в установщик не входят, кладёт их
    «Инструменты разработки/install_dictors_11a.py»."""
    from .f5 import default_voices_dir
    return os.path.join(default_voices_dir(), "qwen")


def trained_model(voice: str) -> str | None:
    """Обученный голос (11а, шаг 2): рядом с образцом своего голоса - «<имя>.model»
    с папкой модели в models_dir() (ru-stress + адаптер чтеца, lora_to_mlx_11a.py).
    Такой голос звучит без чтения образца (только вектор голоса, как в обучении):
    дрейфа ICL нет, голос один на всю книгу."""
    f = os.path.join(own_voices_dir(), voice + ".model")
    if not os.path.isfile(f):
        return None
    name = open(f, encoding="utf-8").read().strip()
    path = name if os.path.isabs(name) else os.path.join(models_dir(), name)
    return path if os.path.exists(os.path.join(path, "config.json")) else None


def trained_gen(voice: str) -> dict:
    """Свои настройки выборки обученного голоса - «<имя>.gen.json» рядом с образцом
    ({"temperature": 0.8, "top_p": 0.85, "top_k": 30}; "icl": true - с образцом и его
    расшифровкой, как клон, а не только вектор голоса): обученной модели нужна выборка
    спокойнее, чем клону (замер 21.10: слов 88 -> 91 %), а общий профиль Qwen
    («Диктор 1-3», оценка Андрея 5 из 5) не трогаем."""
    f = os.path.join(own_voices_dir(), voice + ".gen.json")
    try:
        return json.load(open(f, encoding="utf-8")) if os.path.isfile(f) else {}
    except (OSError, ValueError):
        return {}


def own_voices() -> list[str]:
    d = own_voices_dir()
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d)
                  if f.endswith(".wav") and os.path.isfile(os.path.join(d, f[:-4] + ".txt")))


_VOW = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def strip_short(text: str) -> str:
    """Без знака ударения у односложных слов: «н+а борт+у» -> «на борт+у». Два знака
    подряд («на́ борту́») ru-stress сдвигает («на бАрту»); замер 3.9 на 50 трудных
    словах: 45 -> 48 верных по звуку (meter2)."""
    import re
    return re.sub(r"[А-Яа-яЁё+]+",
                  lambda m: m.group(0).replace("+", "") if sum(c in _VOW for c in m.group(0)) <= 1
                  else m.group(0), text)


def plus_to_acute(text: str) -> str:
    """«з+амок» -> «за́мок»: U+0301 после ударной гласной (так учили ru-stress);
    у односложных слов знака нет (strip_short)."""
    from ..stress import to_combining
    return to_combining(strip_short(text))


class MlxAudioEngine(TTSEngine):
    """Клон через модель mlx-audio. Говорит с отдельным процессом."""
    name = "mlxaudio"
    supports_cloning = True
    stress_format = "+"              # пайплайн ставит «+»; Qwen получает U+0301
    max_chunk_chars = 300

    def __init__(self, model: str = "qwen-stress", bits: int = 8, language: str = "ru",
                 seed: int | None = None, centroid: str | None = None,
                 temperature: float | None = None, top_p: float | None = None,
                 max_chars: int | None = None, **_kw) -> None:
        super().__init__()
        self.model = model
        self.bits = bits
        self.language = language
        self.seed = seed
        self.sample_rate = 24000
        self._proc = None
        self._loaded_model = None         # папка модели, загруженной в процесс
        self._lock = threading.Lock()
        self.last_stats: dict = {}
        # 3.9: одно зерно на всю книгу (голос ровнее); повтор того же текста (пересчёт)
        # - следующее зерно, иначе пересчёт дал бы тот же звук
        self._calls: dict[str, int] = {}
        self.centroid = centroid          # папка фрагментов чтеца - средний вектор голоса
        if max_chars:                     # длиннее кусок - живее интонация (замер 3.9)
            self.max_chunk_chars = int(max_chars)
        self.extra = {k: v for k, v in (("temperature", temperature), ("top_p", top_p)) if v is not None}
        if MODELS[model]["stress"] is None:
            self.stress_format = None     # «+» модель прочитала бы вслух

    # --- голоса: свои голоса модели («Диктор 1-3», 11а), затем образцы F5 ---
    def list_voices(self) -> list[Voice]:
        from .f5 import F5Engine
        own = [Voice(n, n, "", "ru") for n in own_voices()]
        return own + [v for v in F5Engine().list_voices() if v.id not in set(own_voices())]

    def cache_key(self) -> str:
        return (f"mlxaudio|{self.model}|{model_path(self.model, self.bits)}|{self.bits}|"
                f"{self.seed}|{self.centroid}|{sorted(self.extra.items())}")

    # --- процесс ---
    def _call(self, d: dict) -> dict:
        with self._lock:
            self._proc.stdin.write(json.dumps(d, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("процесс mlx-audio завершился")
                if line.startswith("{"):
                    break                      # посторонние строки библиотек пропускаем
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
        self._switch(model_path(self.model, self.bits))

    def _switch(self, path: str) -> None:
        """Загрузить в процесс модель path (обученный голос - своя модель)."""
        if self._loaded_model == path:
            return
        r = self._call({"op": "load", "model": path})
        self._loaded_model = path
        self.sample_rate = int(r["sr"])
        self.load_stats = r
        if self.centroid and path == model_path(self.model, self.bits):
            self._call({"op": "centroid", "dir": self.centroid})

    def close(self) -> None:
        p, self._proc = self._proc, None
        self._loaded_model = None
        if p is not None:
            try:
                p.stdin.write('{"op": "quit"}\n')
                p.stdin.flush()
                p.wait(timeout=10)
            except Exception:
                p.kill()
        self._loaded = False

    def _lang(self) -> str:
        """Язык для модели: у Qwen и VoxCPM - слово, у Chatterbox - код."""
        if self.model == "chatterbox":
            return self.language or "ru"
        return {"ru": "Russian", "en": "English"}.get(self.language, MODELS[self.model]["lang"])

    def _ref(self, voice: str) -> tuple[str, str]:
        from .f5 import default_voices_dir
        if voice.endswith(".wav") and os.path.isabs(voice):        # образец по пути (3.9)
            wav, txt = voice, voice[:-4] + ".txt"
        else:
            d = own_voices_dir() if voice in own_voices() else default_voices_dir()
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
        lang = self._lang()
        trained = trained_model(voice) if self.model == "qwen-stress" else None
        self._switch(trained or model_path(self.model, self.bits))
        extra = self.extra
        if trained:
            g = trained_gen(voice)
            if not g.get("icl"):              # как в обучении: вектор голоса, без языка
                ref_text, lang = None, "auto"
            extra = {**self.extra, **{k: v for k, v in g.items() if k != "icl"}}
        out = tempfile.mktemp(suffix=".wav", prefix="herald_mlxa_")
        try:
            if seed is None and self.seed is not None:
                k = self._calls.get(text, 0)
                self._calls[text] = k + 1
                seed = self.seed + 1000 * k
            r = self._call({"op": "synth", "text": text, "out": out, "ref_audio": wav,
                            "ref_text": ref_text, "lang": lang, "seed": seed,
                            "stream": stream, "extra": extra})
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
