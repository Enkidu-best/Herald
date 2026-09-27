"""Движок синтеза на F5-TTS (русская модель accent_tune) с клонированием голоса.

Даёт естественный, «богатый» голос и слушается ударений (формат '+' перед
ударной гласной - их ставит модуль stress через RUAccent). Голос задаётся
образцом (референсом): короткий чистый фрагмент речи диктора. Модель и образец
готовятся один раз и переиспользуются локально и бесплатно.

Модель: Misha24-10/F5-TTS_RUSSIAN, вариант F5TTS_v1_Base_accent_tune.
Работает на Apple Silicon (MPS/CPU). Для книг целиком на M-чипе позже можно
перевести на MLX ради скорости - голос тот же.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
import numpy as np

# на MPS часть операций F5 не реализована — разрешаем откат на CPU (иначе падение)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .base import TTSEngine, Voice

HF_REPO = "Misha24-10/F5-TTS_RUSSIAN"
CKPT_REL = "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors"
VOCAB_REL = "F5TTS_v1_Base/vocab.txt"

REF_SECONDS = 12          # верхняя граница образца: F5 всё равно клипует ~12 с
REF_MIN_SECONDS = 8       # короче — модели не хватает материала на тембр
PROBE_SECONDS = 25        # столько слушаем в каждой пробной точке записи
START_BONUS = 12.0        # фора началу записи при выборе образца
WHISPER_MLX = "mlx-community/whisper-large-v3-turbo"   # быстрый и точный на M-чипе
WHISPER_FALLBACK = "small"                              # если MLX недоступен


def default_voices_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "text2audio", "voices")
    os.makedirs(d, exist_ok=True)
    return d


class F5Engine(TTSEngine):
    name = "f5"
    sample_rate = 24000
    max_chunk_chars = 200          # F5 стабильнее на коротких кусках
    supports_cloning = True
    stress_format = "+"            # accent_tune понимает '+' перед ударной гласной

    def __init__(self, voices_dir: str | None = None, device: str | None = "cpu",
                 nfe_step: int = 32, speed: float = 1.55) -> None:
        super().__init__()
        self.voices_dir = voices_dir or default_voices_dir()
        # По умолчанию CPU: F5 на MPS (видеопамять Apple Silicon) переполняет
        # память и роняет процесс (segfault). CPU медленнее, но стабильно.
        # Быстрый путь на M-чипе — отдельная MLX-версия (следующий шаг).
        self.device = device
        self.nfe_step = nfe_step        # меньше = быстрее генерация, чуть ниже качество
        # 1.0 = ровно темп образца, его и слышно на выходе (26,8 байт/с при
        # образце в 28,9). Раньше стояло 1.55 — этим компенсировали
        # разрежённый образец (заголовок с паузами, ~14 байт/с вместо ~30).
        # С нормальным образцом ускорять не нужно: на 1.0 выходит 29.6 байт/с
        # против 30.9 у живого диктора.
        self.speed = speed
        self._f5 = None
        self._current = None            # (voice_id, wav_path, ref_text)

    # --- голоса = подготовленные образцы --------------------------------
    def list_voices(self) -> list[Voice]:
        out = []
        for fn in sorted(os.listdir(self.voices_dir)):
            if fn.lower().endswith((".wav", ".flac")):
                stem = os.path.splitext(fn)[0]
                if os.path.isfile(os.path.join(self.voices_dir, stem + ".txt")):
                    out.append(Voice(stem, stem, "", "ru"))
        return out

    # --- загрузка модели ------------------------------------------------
    def load(self, progress=None) -> None:
        import os as _os
        try:
            import torch
            torch.set_num_threads(_os.cpu_count() or 4)
        except Exception:
            pass
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(HF_REPO, CKPT_REL)
        vocab = hf_hub_download(HF_REPO, VOCAB_REL)
        from f5_tts.api import F5TTS
        kw = dict(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab)
        if self.device:
            kw["device"] = self.device
        self._f5 = F5TTS(**kw)
        self._loaded = True

    # --- подготовка образца (разово на голос) ---------------------------
    def prepare_reference(self, audio_path: str, name: str | None = None,
                          ref_text: str | None = None,
                          start: float | None = None,
                          seconds: float = REF_SECONDS) -> str:
        """Выбрать из записи лучший кусок речи, расшифровать и сохранить как голос.

        Что здесь важно и почему (это напрямую определяет качество клона):

        1. Кусок берём НЕ с начала файла. В начале главы диктор читает заголовок
           с паузами — на таком образце получается ~14 байт текста на секунду
           вместо обычных ~28. F5 считает длительность по соотношению
           «байт текста / секунда образца», поэтому вдвое разрежённый образец
           заставляет модель вдвое растягивать речь. Раньше это компенсировали
           темпом 1.55, и голос звучал беднее оригинала.
        2. Границы куска берём по границам фраз из расшифровки, а не «первые
           10 секунд»: иначе образец начинается с обрезанного слова, а его текст
           этому слову не соответствует.
        3. Расшифровка — whisper large-v3-turbo на MLX: быстрая (секунды) и
           точная. Модель small врала («раздерающимскрежитан» вместо
           «душераздирающим скрежетом»), а неверный текст образца портит и голос.

        Возвращает voice_id (имя голоса), который дальше передаётся как voice.
        """
        name = name or os.path.splitext(os.path.basename(audio_path))[0]
        name = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "voice"
        wav_out = os.path.join(self.voices_dir, name + ".wav")
        txt_out = os.path.join(self.voices_dir, name + ".txt")

        if start is not None:
            # ручной режим: кусок задан человеком (подача диктора в начале главы
            # бывает «наряднее», чем в середине, и это слышно)
            self._cut(audio_path, start, seconds, wav_out)
            ref_text = ref_text or self._transcribe_text(wav_out)
        elif ref_text:
            self._cut(audio_path, 0, seconds, wav_out)
        else:
            start, dur, ref_text = self._pick_best_excerpt(audio_path)
            self._cut(audio_path, start, dur, wav_out)

        ref_text = (ref_text or "").strip()
        if ref_text and ref_text[-1] not in ".!?…":
            ref_text += "."
        open(txt_out, "w").write(ref_text)
        return name

    @staticmethod
    def _cut(src: str, start: float, dur: float, dst: str) -> None:
        ff = shutil.which("ffmpeg") or "ffmpeg"
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
                        "-ac", "1", "-ar", "24000", dst], check=True)

    def _pick_best_excerpt(self, audio_path: str) -> tuple[float, float, str]:
        """(начало, длительность, текст) самого подходящего куска записи.

        Главный критерий — ЧИСТОТА ФОНА, а не плотность речи. F5 клонирует не
        только тембр, но и акустическую обстановку образца: если в паузах
        образца шумит, то и синтез будет шуметь. Замерено на этой аудиокниге —
        отношение сигнал/шум переносится в выход почти один в один:

            образец 18,9 дБ  ->  синтез 18,9 дБ   (слышны «скрипы», звук
                                                   металлический и плоский)
            образец 69,2 дБ  ->  синтез 76,0 дБ   (чисто)

        Поэтому сначала отбираем тихий фон, и только потом — беглую речь.
        """
        import numpy as np
        import soundfile as sf

        total = self._duration(audio_path)
        # НАЧАЛО ЗАПИСИ ИДЁТ ПЕРВЫМ И С ФОРОЙ. В начале главы диктор обычно
        # читает ровнее и «наряднее», а фон там чище всего (часто буквально
        # цифровая тишина в паузах). Именно такой образец слушатель отобрал как
        # лучший. Если начало не годится (музыка, шум), сработает общий поиск.
        step = max(PROBE_SECONDS, total / 24)
        probes = [0.0] + [t for t in np.arange(total * 0.05, total * 0.92, step)]
        probes = [p for p in probes if p + PROBE_SECONDS < total] or [0.0]

        best = None
        tmp = os.path.join(tempfile.gettempdir(), f"t2a_probe_{os.getpid()}.wav")
        for start in probes:
            try:
                self._cut(audio_path, float(start), PROBE_SECONDS, tmp)
                segs = self._transcribe_segments(tmp)
                picked = self._span(segs)
                if not picked:
                    continue
                t0, t1, text = picked
                a, sr = sf.read(tmp, start=int(t0 * 24000), stop=int(t1 * 24000))
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                if not len(a):
                    continue
                sec = len(a) / sr
                rms = float(np.sqrt(np.mean(a ** 2)))
                snr = self._snr(a, sr, rms)
                bps = len(text.encode("utf-8")) / sec
                # чистота решает; темп речи — уточняющая поправка (ориентир ~29 байт/с,
                # это обычная скорость чтения вслух), громкость — совсем мелкая
                score = min(snr, 70.0) - 2.0 * abs(bps - 29.0) - 20.0 * abs(rms - 0.12)
                if start < 1.0:
                    score += START_BONUS      # фора началу записи
                if best is None or score > best[0]:
                    best = (score, start + t0, sec, text, bps, rms, snr)
            except Exception:
                continue
        if os.path.exists(tmp):
            os.remove(tmp)

        if best is None:                       # запись короткая или нераспознаваемая
            return 0.0, min(REF_SECONDS, total), self._transcribe_text(audio_path)
        _sc, start, sec, text, bps, rms, snr = best
        print(f"[голос] взят кусок {start:.1f}-{start+sec:.1f} c "
              f"(фон {snr:.0f} дБ, {bps:.0f} байт/с, громкость {rms:.3f})",
              file=sys.stderr)
        return start, sec, text

    @staticmethod
    def _snr(a, sr: int, rms: float) -> float:
        """Насколько речь громче фона. Фон — уровень в самых тихих окнах."""
        import numpy as np
        win = int(0.05 * sr)
        n = len(a) // win
        if n < 4:
            return 0.0
        e = np.array([np.sqrt(np.mean(a[i * win:(i + 1) * win] ** 2)) for i in range(n)])
        floor = float(np.percentile(e, 5))
        return float(20 * np.log10(rms / max(floor, 1e-12)))

    @staticmethod
    def _span(segs: list[tuple[float, float, str]]) -> tuple[float, float, str] | None:
        """Набрать подряд идущие фразы на 8-12 секунд, не разрывая слов."""
        for i in range(len(segs)):
            t0 = segs[i][0]
            text_parts = []
            for j in range(i, len(segs)):
                t1 = segs[j][1]
                text_parts.append(segs[j][2])
                if t1 - t0 >= REF_MIN_SECONDS:
                    if t1 - t0 <= REF_SECONDS:
                        return t0, t1, " ".join(x.strip() for x in text_parts).strip()
                    break
        return None

    def _transcribe_segments(self, wav_path: str) -> list[tuple[float, float, str]]:
        """Фразы с таймкодами. Сначала быстрый MLX-turbo, иначе faster-whisper."""
        try:
            import mlx_whisper
            r = mlx_whisper.transcribe(wav_path, path_or_hf_repo=WHISPER_MLX,
                                       language="ru", verbose=False)
            return [(s["start"], s["end"], s["text"]) for s in r["segments"]]
        except Exception:
            from faster_whisper import WhisperModel
            m = WhisperModel(WHISPER_FALLBACK, device="cpu", compute_type="int8")
            segs, _ = m.transcribe(wav_path, language="ru")
            return [(s.start, s.end, s.text) for s in segs]

    def _transcribe_text(self, wav_path: str) -> str:
        try:
            return " ".join(t for _a, _b, t in self._transcribe_segments(wav_path)).strip()
        except Exception:
            return ""

    @staticmethod
    def _duration(path: str) -> float:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True).stdout.strip()
        try:
            return float(out)
        except ValueError:
            return 0.0

    def _resolve_voice(self, voice: str):
        # voice может быть id голоса или путём к образцу
        if os.path.isfile(voice):
            vid = self.prepare_reference(voice)
        else:
            vid = voice
        wav = os.path.join(self.voices_dir, vid + ".wav")
        txt = os.path.join(self.voices_dir, vid + ".txt")
        if not os.path.isfile(wav):
            raise ValueError(f"Образец голоса не найден: {vid}. Сначала загрузите образец.")
        ref_text = open(txt).read().strip() if os.path.isfile(txt) else ""
        return vid, wav, ref_text

    # --- синтез ---------------------------------------------------------
    def synth_chunk(self, text: str, voice: str, **kw) -> np.ndarray:
        self.ensure_loaded()
        if not self.has_speech(text):
            return np.zeros(0, dtype=np.float32)
        if self._current is None or self._current[0] not in (voice,
                                                             os.path.splitext(os.path.basename(voice))[0]):
            self._current = self._resolve_voice(voice)
        _vid, ref_wav, ref_text = self._current
        try:
            # seed задаём сами и строго в допустимом диапазоне: F5TTS.infer без него
            # берёт random.randint(0, sys.maxsize) и кладёт это в PYTHONHASHSEED, а
            # там разрешено только 0..4294967295. После такого любой дочерний
            # python падает с «Fatal Python error: config_init_hash_seed» (ломался
            # вспомогательный процесс multiprocessing, в консоль летела ошибка).
            wav, sr, _ = self._f5.infer(
                ref_file=ref_wav, ref_text=ref_text, gen_text=text.strip(),
                nfe_step=self.nfe_step, speed=self.speed, remove_silence=False,
                seed=random.randrange(2**32),
            )
            # wav может прийти как torch-тензор (в т.ч. на MPS) — приводим к numpy надёжно
            if hasattr(wav, "detach"):
                wav = wav.detach().to("cpu").numpy()
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            # защита от NaN/Inf (иногда даёт MPS) — иначе весь файл «тишина»
            if not np.all(np.isfinite(wav)):
                wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            self._empty_cache()
            return wav
        except Exception as e:
            import sys
            print(f"[F5] пропущен кусок: {e}", file=sys.stderr)
            self._empty_cache()
            return np.zeros(0, dtype=np.float32)

    def _empty_cache(self):
        # на MPS чистим кэш видеопамяти между кусками — иначе память копится и падает
        if self.device and "mps" in str(self.device):
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
