"""Сборка озвученных глав в аудиокнигу через ffmpeg.

Выход:
  - книга целиком в .m4b с главами-закладками, обложкой и метаданными
    (её понимают Apple Books / Books, листается по главам);
  - при желании — отдельные файлы по главам (.mp3).

ffmpeg берётся из системы (у Scribe он уже стоит через brew).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class RenderedChapter:
    title: str
    wav_path: str
    duration: float          # секунды


# Полировка речи перед кодированием. Подбиралась не «на глаз», а по спектру
# того варианта, который слушатель отобрал как лучший, — цель и результат:
#
#   полоса        нравится   получилось
#   60-250 Гц     12,8%      12,3%     ← бас; его нельзя срезать, иначе «плоско»
#   250-700 Гц    36,4%      33,9%
#   2,5-6 кГц     20,4%      21,9%     ← присутствие, разборчивость
#   выше 6 кГц    11,6%      11,7%     ← «воздух»
#
# Прошлая версия цепочки (highpass 80 Гц и -2,5 дБ на 350 Гц из заметок по
# демо-клипам) роняла бас с 12,8% до 7,1% — звук становился тонким и плоским.
# Здесь низ только чистится от совсем уж инфранизкого гула и слегка поднимается.
# loudnorm убран намеренно: он ужимал и без того сжатую динамику синтеза.
# Про «металл»: выше ~10 кГц у F5 уже не речь, а то, что дорисовал вокодер.
# Полка «+2,5 дБ от 9 кГц» усиливала именно этот мусор. Теперь верх поднимаем
# узко на 7,5 кГц (там ещё настоящие согласные), а выше 11 кГц срезаем:
# по замеру это убирает треть энергии в полосе 10-12 кГц и почти всю выше 12.
HD_FILTER = (
    "highpass=f=50,"                              # только инфранизкий гул
    "equalizer=f=110:t=q:w=1.0:g=3.5,"            # тело и бас голоса
    "equalizer=f=450:t=q:w=1.4:g=-1,"             # чуть убрать «бочку»
    "equalizer=f=3500:t=q:w=1.5:g=2,"             # присутствие согласных
    "equalizer=f=7500:t=q:w=1.4:g=2,"             # «воздух» там, где он настоящий
    "lowpass=f=11000,"                            # срез вокодерных артефактов
    "volume=4dB,"
    "alimiter=limit=0.95,"                        # без клиппинга, но и без сжатия
    "aresample=48000"
)


def write_wav(audio: np.ndarray, sr: int, path: str) -> float:
    sf.write(path, audio, sr)
    return len(audio) / sr if len(audio) else 0.0


def _ff() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg не найден. Установите: brew install ffmpeg")
    return exe


def _esc_meta(v: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        v = v.replace(ch, "\\" + ch)
    return v


def _ffmetadata(chapters: list[RenderedChapter], title: str, author: str) -> str:
    lines = [";FFMETADATA1"]
    if title:
        lines.append(f"title={_esc_meta(title)}")
    if author:
        lines.append(f"artist={_esc_meta(author)}")
        lines.append(f"album={_esc_meta(title or author)}")
    lines.append("genre=Audiobook")
    t = 0
    for ch in chapters:
        start = int(round(t * 1000))
        t += ch.duration
        end = int(round(t * 1000))
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start}")
        lines.append(f"END={end}")
        lines.append(f"title={_esc_meta(ch.title)}")
    return "\n".join(lines) + "\n"


def make_title_cover(title: str, author: str, path: str) -> str:
    """Простая карточка-обложка, когда в файле обложки нет."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1400, 2100
    img = Image.new("RGB", (W, H), (33, 37, 43))
    d = ImageDraw.Draw(img)

    def font(size):
        for name in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                     "/System/Library/Fonts/Helvetica.ttc",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.isfile(name):
                try:
                    return ImageFont.truetype(name, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def wrap(text, fnt, max_w):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if d.textlength(trial, font=fnt) <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    d.rectangle([60, 60, W - 60, H - 60], outline=(90, 98, 110), width=4)
    tf, af = font(96), font(56)
    y = 620
    for ln in wrap(title or "Без названия", tf, W - 260):
        d.text((130, y), ln, font=tf, fill=(240, 242, 245))
        y += 120
    if author:
        y += 60
        for ln in wrap(author, af, W - 260):
            d.text((130, y), ln, font=af, fill=(170, 178, 190))
            y += 74
    img.save(path, "JPEG", quality=88)
    return path


def _prepare_cover(cover_bytes: bytes | None, cover_ext: str, title: str,
                   author: str, workdir: str) -> str:
    """Вернуть путь к jpg-обложке (из файла книги или сгенерированной)."""
    out = os.path.join(workdir, "cover.jpg")
    if cover_bytes:
        try:
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(cover_bytes)).convert("RGB")
            im.save(out, "JPEG", quality=88)
            return out
        except Exception:
            pass
    return make_title_cover(title, author, out)


def assemble_m4b(chapters: list[RenderedChapter], out_path: str, *,
                 title: str = "", author: str = "",
                 cover_bytes: bytes | None = None, cover_ext: str = "jpg",
                 bitrate: str = "96k") -> str:
    """Склеить главы в один .m4b с закладками и обложкой."""
    ff = _ff()
    workdir = tempfile.mkdtemp(prefix="t2a_")
    try:
        # список для concat-демуксера
        listfile = os.path.join(workdir, "list.txt")
        with open(listfile, "w") as f:
            for ch in chapters:
                f.write(f"file '{ch.wav_path}'\n")
        metafile = os.path.join(workdir, "meta.txt")
        with open(metafile, "w") as f:
            f.write(_ffmetadata(chapters, title, author))
        cover = _prepare_cover(cover_bytes, cover_ext, title, author, workdir)

        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", listfile,
            "-i", metafile,
            "-i", cover,
            "-map_metadata", "1",
            "-map", "0:a", "-map", "2:v",
            "-c:a", "aac", "-b:a", bitrate,
            "-c:v", "mjpeg", "-disposition:v", "attached_pic",
            "-dn",                       # без служебных data-потоков
            "-movflags", "+faststart",
            "-f", "mp4", out_path,
        ]
        subprocess.run(cmd, check=True)
        return out_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# Форматы вывода. Слушатель сравнил wav / mp3 320 / mp3 192 и разницы между
# 320 и 192 не услышал, поэтому 192 — разумная база: 8-часовая книга займёт
# около 660 МБ вместо 2,6 ГБ у wav. Кому нужен предел — FLAC (без потерь).
FORMATS: dict[str, tuple[str, list[str]]] = {
    "mp3 192 кбит/с (обычный)": (".mp3", ["-codec:a", "libmp3lame", "-b:a", "192k",
                                          "-write_xing", "1"]),
    "mp3 320 кбит/с":           (".mp3", ["-codec:a", "libmp3lame", "-b:a", "320k",
                                          "-write_xing", "1"]),
    "FLAC (без потерь)":        (".flac", ["-codec:a", "flac", "-compression_level", "5"]),
}
DEFAULT_FORMAT = "mp3 192 кбит/с (обычный)"


def encode_chapter_mp3(ch: RenderedChapter, out_path: str, *,
                       title: str = "", author: str = "", quality: str = "2",
                       hd: bool = True, fmt: str = DEFAULT_FORMAT) -> str:
    ff = _ff()
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", ch.wav_path,
    ]
    if hd:
        cmd += ["-af", HD_FILTER]
    _ext, codec = FORMATS.get(fmt, FORMATS[DEFAULT_FORMAT])
    cmd += codec + [
        "-metadata", f"title={ch.title}",
        "-metadata", f"artist={author}",
        "-metadata", f"album={title}",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def polish_wav_to_mp3(wav_path: str, out_path: str, *, hd: bool = True,
                      quality: str = "2", fmt: str = DEFAULT_FORMAT) -> str:
    """Тот же путь, что у глав, но для одиночного файла (пробник).

    Пробник должен звучать ровно так же, как итоговая книга, иначе по нему
    нельзя судить о результате.
    """
    ff = _ff()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path]
    if hd:
        cmd += ["-af", HD_FILTER]
    _ext, codec = FORMATS.get(fmt, FORMATS[DEFAULT_FORMAT])
    cmd += codec + [out_path]
    subprocess.run(cmd, check=True)
    return out_path
