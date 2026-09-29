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
MIN_REF_SECONDS = 7       # короче образец — клон теряет устойчивость тембра
# Сколько звука отдаём распознаванию при отборе образца. Расшифровка идёт со
# скоростью около 8x реального времени, и на двадцатиминутной записи она заняла
# 2,5 минуты — на книге в десять часов это были бы часы. Поэтому сначала дешёвый
# предотбор по тембру (без распознавания вообще), и расшифровываются только
# перспективные участки.
AUDIT_BUDGET_SEC = 420
REGION_PAD = 25           # сколько звука берём вокруг перспективного окна
REF_MIN_SECONDS = 8       # короче — модели не хватает материала на тембр
PROBE_SECONDS = 25        # столько слушаем в каждой пробной точке записи
START_BONUS = 6.0         # небольшая фора началу записи при выборе образца
TEMBRE_W = 25.0           # вес совпадения тембра образца со средним по записи
TEMBRE_CLEAN = 30.0       # штраф за верх ГРЯЗНЕЕ, чем в среднем по записи
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
                 nfe_step: int = 32, speed: float = 1.0) -> None:
        super().__init__()
        self.voices_dir = voices_dir or default_voices_dir()
        # По умолчанию CPU: F5 на MPS (видеопамять Apple Silicon) переполняет
        # память и роняет процесс (segfault). CPU медленнее, но стабильно.
        # Быстрый путь на M-чипе — отдельная MLX-версия (следующий шаг).
        self.device = device
        self.nfe_step = nfe_step        # меньше = быстрее генерация, чуть ниже качество
        # 1.0 = «обычный» темп. Абсолютную скорость движок подгоняет под
        # образец сам (target_bps), поэтому здесь просто множитель.
        # Раньше стояло 1.55 — этим компенсировали
        # разрежённый образец (заголовок с паузами, ~14 байт/с вместо ~30).
        # С нормальным образцом ускорять не нужно: на 1.0 выходит 29.6 байт/с
        # против 30.9 у живого диктора.
        self.speed = speed
        self._f5 = None
        self._current = None            # (voice_id, wav_path, ref_text)

    # --- голоса = подготовленные образцы --------------------------------
    def list_voices(self) -> list[Voice]:
        """Голоса с пометкой языка.

        Язык определяем по расшифровке образца: на каком языке говорил диктор,
        для такого языка голос и годится. Английскую книгу русским образцом
        озвучивать бессмысленно — модели разные, выйдет тяжёлый акцент.
        """
        from ..normalize import detect_language
        out = []
        for fn in sorted(os.listdir(self.voices_dir)):
            if not fn.lower().endswith((".wav", ".flac")):
                continue
            stem = os.path.splitext(fn)[0]
            txt = os.path.join(self.voices_dir, stem + ".txt")
            if not os.path.isfile(txt):
                continue
            try:
                lang = detect_language(open(txt, encoding="utf-8").read())
            except Exception:
                lang = "ru"
            out.append(Voice(stem, stem, "", lang))
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
        self._save_source(name, audio_path)
        return name

    def _save_source(self, name: str, audio_path: str) -> None:
        """Запомнить, из какой записи сделан голос, и кто в ней читает.

        Без этого невозможно потом сверить результат с оригиналом: имя голоса
        ничего не говорит о файле. Один раз я на этом и ошибся — сравнивал
        синтез одного чтеца с записью другого и делал выводы по чужим числам.
        Имя чтеца берём из ID3 (`album_artist`, у аудиокниг там именно чтец).
        """
        import json
        info = {"source": os.path.abspath(audio_path)}
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format_tags=album_artist,artist,album,title",
                 "-of", "default=nw=1", audio_path],
                capture_output=True, text=True).stdout
            tags = dict(line.split("=", 1) for line in out.strip().splitlines()
                        if "=" in line)
            # У аудиокниг в album_artist обычно чтец — но не всегда: в «Олесе»
            # там стоит Куприн, то есть автор. Если album_artist совпадает с
            # artist, это автор книги, а не чтец, и записывать его как чтеца
            # нельзя — эталон для сверки потом найдут по нему.
            reader = (tags.get("TAG:album_artist") or "").strip()
            author = (tags.get("TAG:artist") or "").strip()
            if reader and reader.casefold() != author.casefold():
                info["reader"] = reader
            book = tags.get("TAG:title") or tags.get("TAG:album") or ""
            if book:
                info["book"] = book.strip()
        except Exception:
            pass
        try:
            with open(os.path.join(self.voices_dir, name + ".src.json"),
                      "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _tembre(y, sr: int):
        """Тембр куска: яркость, «воздух», металл, шипение.

        Считаем по долгому среднему спектру (LTAS) и только по кадрам с речью:
        паузы иначе тянут картину вниз, а медиана по кадрам, наоборот, почти не
        различает куски — у неё всё усредняется до типичной гласной.
        """
        import numpy as np
        n = 2048
        hop = n // 2
        if len(y) < n * 4:
            return None
        w = np.hanning(n)
        f = np.fft.rfftfreq(n, 1 / sr)
        top = sr / 2 - 100
        air_m = (f >= 5000) & (f < top)
        met_m = (f >= 4000) & (f < min(11000.0, top))
        sib_m = (f >= 5000) & (f < min(9000.0, top))
        ref_m = (f >= 1000) & (f < 4000)
        if met_m.sum() < 4:
            return None
        y = np.asarray(y, dtype=np.float64)
        pk = float(np.max(np.abs(y))) + 1e-9
        # Клиппинг: срезанные вершины волны, три отсчёта подряд у самого верха.
        # В образце это искажение, и клон его наследует. У английского исходника
        # таких мест 417 — кусок оттуда брать нельзя.
        flat = np.abs(y) > 0.985 * pk
        clipped = float(np.mean(flat[2:] & flat[1:-1] & flat[:-2]))
        y = y / pk
        frames = [y[i:i + n] for i in range(0, len(y) - n, hop)]
        rms = np.array([np.sqrt(np.mean(fr ** 2)) for fr in frames])
        keep = rms > max(float(np.percentile(rms, 55)), 1e-5)
        if keep.sum() < 4:
            return None
        acc = np.zeros(n // 2 + 1)
        for fr, ok in zip(frames, keep):
            if ok:
                acc += np.abs(np.fft.rfft(fr * w)) ** 2
        acc /= max(int(keep.sum()), 1)
        tot = acc.sum() + 1e-12
        S = np.sqrt(acc)
        b = acc[met_m] + 1e-10
        # Порог только относительный. С абсолютной добавкой (0.06) метрика
        # зависела от громкости куска: после нормировки по пику тихий кусок
        # «трещал» вдвое сильнее громкого при том же качестве, и отбор браковал
        # лучшие места записи.
        d = np.abs(np.diff(y))
        clicks = int(np.sum(d > 25.0 * (float(np.median(d)) + 1e-9)))
        # «Свист» — узкий тон в верху. Мерим выступ над ОКРЕСТНОСТЬЮ +-400 Гц, а
        # не над медианой полосы: медиана зависит от того, докуда вообще
        # простирается запись, и у файла с частотой 22 кГц пустой верх давал
        # фантастические 100 дБ там, где никакого тона нет. У живой речи выступ
        # 5-12 дБ, у синтеза со свистом — 18 дБ и выше.
        band = np.where((f >= 3500) & (f < min(11000.0, top)))[0]
        tone = 0.0
        for j in band[::4]:
            nb = (f >= f[j] - 400) & (f <= f[j] + 400)
            d = (10.0 * np.log10(acc[j] + 1e-20)
                 - 10.0 * np.log10(float(np.median(acc[nb])) + 1e-20))
            tone = max(tone, float(d))
        # Докуда простирается верх. Если запись обрезана заметно ниже 12 кГц (это
        # потолок F5), верхние полосы образца пустые, вокодер их домысливает — и
        # добавляет свист: у образца из записи 32 кбит/с (верх до 9 кГц) синтез
        # дал выступ 17,7 дБ против 5,3 дБ у образца из полноценной записи.
        mid_lvl = float(acc[(f >= 1000) & (f < 3000)].mean()) + 1e-20
        above = np.where(acc > mid_lvl * 1e-4)[0]
        ceiling = float(f[above[-1]]) if len(above) else 0.0
        # «Верх» в децибелах к середине голоса, а не долей от всей энергии.
        # Доля обманывает: у записи с плотным низом она мала даже при ярком
        # верхе — «Олеся» по доле выглядела вдвое темнее записи 1, а в дБ
        # оказалась ярче её на 8 дБ в полосе 8-10 кГц.
        hf = (10.0 * np.log10(float(acc[air_m].mean()) + 1e-20)
              - 10.0 * np.log10(float(acc[(f >= 1000) & (f < 3000)].mean()) + 1e-20))
        return {"tone": tone, "ceiling": ceiling, "hf": hf, "clipped": clipped,
                "air": float(100.0 * acc[air_m].sum() / tot),
                "centroid": float((f * S).sum() / (S.sum() + 1e-9)),
                "metal": float(np.exp(np.mean(np.log(b))) / np.mean(b)),
                "sibilance": float(acc[sib_m].sum() / (acc[ref_m].sum() + 1e-9)),
                "crackle": float(clicks / (len(y) / sr))}

    def _scan(self, audio_path: str, total: float, sr: int = 24000):
        """[(секунда, признаки)] по всей записи — потоком, не грузя её в память.

        Запись может быть и трёхчасовой книгой: при загрузке целиком это два с
        лишним гигабайта. Поэтому читаем из ffmpeg кусками и считаем признаки на
        лету, а сами отсчёты тут же выбрасываем. Шаг подстраивается под длину,
        чтобы на любой записи выходило около тысячи окон.
        """
        import numpy as np
        win = 10
        hop = max(5, int(total / 1200) + 1)
        ff = shutil.which("ffmpeg") or "ffmpeg"
        proc = subprocess.Popen(
            [ff, "-v", "error", "-i", audio_path, "-ac", "1", "-ar", str(sr),
             "-f", "f32le", "-"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        out: list[tuple[float, dict]] = []
        buf = np.zeros(0, dtype=np.float32)
        t, need, step = 0.0, win * sr, hop * sr
        try:
            while True:
                raw = proc.stdout.read(step * 4)
                if not raw:
                    break
                buf = np.concatenate([buf, np.frombuffer(raw, dtype=np.float32)])
                while len(buf) >= need:
                    seg = buf[:need].astype(np.float64)
                    w = self._tembre(seg, sr)
                    if w:
                        w["rms"] = float(np.sqrt(np.mean(seg ** 2)))
                        out.append((t, w))
                    buf = buf[step:]
                    t += hop
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait()
        return out

    @staticmethod
    def _speech_windows(scan):
        """Окна, где действительно читают.

        На трёхчасовой книге пауз столько, что медиана по ВСЕМ окнам даёт
        профиль тишины: воздух 0,06% и яркость 1361 Гц там, где у речи этого же
        чтеца 2,9% и 2292 Гц. Порог берём от самой записи, а не абсолютный:
        записи бывают и тихие, и громкие.
        """
        import numpy as np
        if not scan:
            return scan
        vals = np.array([w.get("rms", 0.0) for _t, w in scan])
        loud = float(np.percentile(vals, 75))
        keep = [(t, w) for (t, w), v in zip(scan, vals) if v > 0.4 * loud]
        return keep if len(keep) >= max(4, len(scan) // 20) else scan

    @classmethod
    def _profile(cls, scan):
        """Типичный тембр записи — медиана по окнам с речью.

        Медиана, а не максимум: самый звонкий кусок даёт не звонкий голос, а
        поднятый шум вокодера (проверено — металл и шипение выходили выше, чем у
        живого чтеца). И считается она по таким же окнам, какими мерятся
        кандидаты: у одного долгого спектра на всю запись «металл» выходит
        впятеро выше просто из-за усреднения, и штраф за грязный верх не
        срабатывает.
        """
        import numpy as np
        scan = cls._speech_windows(scan)
        if not scan:
            return None
        keys = [k for k in scan[0][1].keys() if k != "rms"]
        return {k: float(np.median([w[k] for _t, w in scan])) for k in keys}

    def _promising_regions(self, scan, target, total: float):
        """Участки записи, похожие по тембру на чтеца, — куда стоит смотреть.

        Предотбор без распознавания: окна оцениваются только по тембру и чистоте
        верха, вокруг лучших берётся по REGION_PAD секунд. Поиск по-прежнему идёт
        по всей записи, но расшифровка достаётся минутам, а не часам: она работает
        со скоростью около 8x реального времени, и на трёхчасовой книге заняла бы
        двадцать с лишним минут.
        """
        if total <= AUDIT_BUDGET_SEC or target is None or not scan:
            return [(0.0, total)]
        scored = []
        for t, c in self._speech_windows(scan):     # в паузы смотреть незачем
            dev = (abs(c["hf"] - target["hf"]) / 6.0
                   + abs(c["centroid"] - target["centroid"])
                   / max(target["centroid"], 200.0))
            dirt = (max(0.0, c["metal"] - target["metal"]) / max(target["metal"], 0.02)
                    + max(0.0, c["sibilance"] - target["sibilance"])
                    / max(target["sibilance"], 0.05)
                    + max(0.0, c["tone"] - target["tone"] - 8.0) / 2.0)
            scored.append((TEMBRE_W * dev + TEMBRE_CLEAN * dirt, t))
        scored.sort()
        regions: list[list[float]] = []
        budget = 0.0
        for _pen, t in scored:
            a, b = max(0.0, t - REGION_PAD), min(total, t + 10 + REGION_PAD)
            for r in regions:
                if a <= r[1] and b >= r[0]:
                    budget += max(0.0, b - r[1]) + max(0.0, r[0] - a)
                    r[0], r[1] = min(r[0], a), max(r[1], b)
                    break
            else:
                regions.append([a, b])
                budget += b - a
            if budget >= AUDIT_BUDGET_SEC:
                break
        regions.sort()
        merged: list[list[float]] = []      # склейка за один проход: участки
        for a, b in regions:                # могли перекрыться между собой
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged]

    @staticmethod
    def _cut(src: str, start: float, dur: float, dst: str) -> None:
        ff = shutil.which("ffmpeg") or "ffmpeg"
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
                        "-ac", "1", "-ar", "24000", dst], check=True)

    def _pick_best_excerpt(self, audio_path: str) -> tuple[float, float, str]:
        """(начало, длительность, текст) самого подходящего куска записи.

        Два критерия, и оба обязательны.

        ЧИСТОТА. F5 клонирует не только тембр, но и акустику образца: шумит в
        паузах образца — шумит и синтез. Отношение сигнал/шум переносится в
        выход почти один в один (замерено: образец 18,9 дБ -> синтез 18,9 дБ,
        «скрипы» и металл; образец 69,2 дБ -> синтез 76,0 дБ, чисто).

        ТЕМБР. Верх, которого в образце нет, потом не вернёт никакой
        эквалайзер — он поднимет только шум вокодера. Прежний отбор смотрел
        лишь на чистоту и взял у чтеца самое глухое место записи: воздух 0,74%
        против 2,70% в среднем по его же голосу.

        Кандидаты — все связки подряд идущих фраз длиной 7-12 с внутри
        перспективных участков.
        """
        import numpy as np
        import soundfile as sf

        total = self._duration(audio_path)
        scan = self._scan(audio_path, total)
        target = self._profile(scan)
        if target:
            print(f"[голос] тембр записи: яркость {target['centroid']:.0f} Гц, "
                  f"воздух {target['air']:.2f}%, металл {target['metal']:.3f}, "
                  f"шипение {target['sibilance']:.3f}, "
                  f"верх до {target['ceiling'] / 1000:.0f} кГц", file=sys.stderr)
            if target["ceiling"] < 10000.0:
                print(f"[голос] ВНИМАНИЕ: верх записи обрезан на "
                      f"{target['ceiling'] / 1000:.0f} кГц (F5 работает до 12). "
                      "Верхние полосы образца пустые, вокодер их домысливает и "
                      "добавляет свист. Запись от 128 кбит/с и 44 кГц даст "
                      "заметно чище.", file=sys.stderr)

        best = None
        part = os.path.join(tempfile.gettempdir(), f"t2a_part_{os.getpid()}.wav")
        try:
            regions = self._promising_regions(scan, target, total)
            if len(regions) > 1:
                print("[голос] смотрим участки: "
                      + ", ".join(f"{a:.0f}-{b:.0f} c" for a, b in regions),
                      file=sys.stderr)
            for r0, r1 in regions:
                if r1 - r0 < MIN_REF_SECONDS:
                    continue
                self._cut(audio_path, r0, r1 - r0, part)
                segs = [(g[0], g[1], g[2]) for g in self._transcribe_segments(part)
                        if (g[2] or "").strip()]
                if not segs:
                    continue
                y, sr = sf.read(part, dtype="float32")
                y = np.asarray(y).reshape(-1)
                # Темп речи сравниваем с темпом ЭТОГО чтеца. Прежний ориентир в
                # 29 байт/с — средняя скорость чтения вслух, но у стихов она
                # вдвое ниже, и тогда все куски записи получали штраф по сорок
                # очков, а выбор решался мелочами.
                spoken = sum(max(g[1] - g[0], 0.01) for g in segs)
                said = sum(len(g[2].encode("utf-8")) for g in segs)
                ref_bps = said / spoken if spoken > 1.0 else 29.0

                cands = []
                for i in range(len(segs)):
                    t0, acc = segs[i][0], []
                    for j in range(i, min(i + 14, len(segs))):
                        t1 = segs[j][1]
                        acc.append(segs[j][2])
                        if t1 - t0 > REF_SECONDS:
                            break
                        if t1 - t0 >= MIN_REF_SECONDS:
                            cands.append((t0, t1, " ".join(acc).strip()))
                stride = max(1, len(cands) // 400)
                for t0, t1, text in cands[::stride]:
                    a = y[int(t0 * sr):int(t1 * sr)]
                    sec = len(a) / sr
                    if sec < MIN_REF_SECONDS:
                        continue
                    rms = float(np.sqrt(np.mean(a ** 2)))
                    if rms < 1e-4:
                        continue
                    snr = self._snr(a, sr, rms)
                    bps = len(text.encode("utf-8")) / sec
                    # чистота решает; темп — поправка, громкость — мелкая.
                    # Плюс предпочтение длине: F5 использует образец целиком, до
                    # ~12 с, и на коротком куске тембр держится хуже. При прочих
                    # равных берём кусок подлиннее.
                    score = (min(snr, 60.0) - 2.0 * abs(bps - ref_bps)
                             - 20.0 * abs(rms - 0.12)
                             + 1.5 * (sec - MIN_REF_SECONDS))
                    cand = self._tembre(a.astype(np.float64), sr)
                    if target and cand:
                        dev = (abs(cand["hf"] - target["hf"]) / 6.0
                               + abs(cand["centroid"] - target["centroid"])
                               / max(target["centroid"], 200.0))
                        # Треск в оценку НЕ входит, хотя и считается для отчёта:
                        # проверено синтезом — треск образца не предсказывает
                        # треск результата, зато отбрасывал лучший кусок записи.
                        dirt = (max(0.0, cand["metal"] - target["metal"])
                                / max(target["metal"], 0.02)
                                + max(0.0, cand["sibilance"] - target["sibilance"])
                                / max(target["sibilance"], 0.05)
                                # свист — брак, а не оттенок тембра
                                + max(0.0, cand["tone"] - target["tone"] - 8.0) / 2.0
                                + 200.0 * cand["clipped"])   # клиппинг — вон
                        score -= TEMBRE_W * dev + TEMBRE_CLEAN * dirt
                    if best is None or score > best[0]:
                        best = (score, t0 + r0, sec, text, bps, rms, snr, cand)
        except Exception as e:
            print(f"[голос] отбор образца не удался ({e}), беру начало записи",
                  file=sys.stderr)
        finally:
            if os.path.exists(part):
                os.remove(part)

        if best is None:                       # запись короткая или нераспознаваемая
            return 0.0, min(REF_SECONDS, total), self._transcribe_text(audio_path)
        _sc, start, sec, text, bps, rms, snr, cand = best
        extra = ""
        if cand:
            extra = (f", яркость {cand['centroid']:.0f} Гц, воздух {cand['air']:.2f}%"
                     f", металл {cand['metal']:.3f}, свист {cand['tone']:.0f} дБ")
        print(f"[голос] взят кусок {start:.1f}-{start+sec:.1f} c "
              f"(фон {snr:.0f} дБ, {bps:.0f} байт/с, громкость {rms:.3f}{extra})",
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
            # язык НЕ навязываем: whisper определяет его сам и делает это точно.
            # Раньше здесь стояло language="ru", и английская запись
            # расшифровывалась русскими буквами («Шоан Антони, е-learning reel»),
            # а с таким текстом образца клон получался никуда не годный.
            r = mlx_whisper.transcribe(wav_path, path_or_hf_repo=WHISPER_MLX,
                                       verbose=False)
            self.detected_language = r.get("language") or "ru"
            return [(s["start"], s["end"], s["text"]) for s in r["segments"]]
        except Exception:
            from faster_whisper import WhisperModel
            m = WhisperModel(WHISPER_FALLBACK, device="cpu", compute_type="int8")
            segs, info = m.transcribe(wav_path)
            self.detected_language = getattr(info, "language", None) or "ru"
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
