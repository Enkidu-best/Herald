"""Быстрые голоса с загрузкой (3.7): каталог, размер на диске, скачать / удалить.

Выбор Андрея по слепым наборам 3.6:
  * русские: Vosk диктор 3 и 4 (муж.), диктор 2 (жен.) - один пакет Vosk на три
    голоса (783 МБ, Apache-2.0); Айдар, Евгений, Ксения, Ксения-2 (Silero) -
    встроенные; Бая скрыта (её можно вернуть в Настройках);
  * английские: Piper joe (муж.), kristin, cori (жен.) - каждый голос отдельно.

Откуда качается - только официальные источники, с проверкой суммы:
  * пакет Vosk - alphacephei.com (md5 из их списка моделей);
  * голоса Piper - huggingface.co/rhasspy/piper-voices (md5 из voices.json);
  * движок Piper (piper-tts, GPL-3.0) - колесо с pypi.org (sha256 из PyPI). В
    установщик он не кладётся (GPL): ставится по кнопке в папку данных и
    подключается оттуда, как ускорение NVIDIA (core/cuda_pack).
Образец звучания «без загрузки»: у Piper - официальный образец с
rhasspy.github.io/piper-samples; у Vosk официальных образцов нет - образец
доступен после загрузки пакета; у Silero - синтез на месте.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

VOSK_MODEL = "vosk-model-tts-ru-0.9-multi"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"
VOSK_MD5 = "2f8b6dbf64e912f9ee7eda50ba2d3c80"
VOSK_SIZE = 782_787_154
PIPER_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
PIPER_SAMPLES = "https://rhasspy.github.io/piper-samples/samples/"
PIPER_VERSION = "1.8.0"
PIPER_LICENSE = ("Движок Piper - GPL-3.0 (скачивается с pypi.org, в программу не входит); "
                 "голос - {lic}. Источник: huggingface.co/rhasspy/piper-voices.")


@dataclass(frozen=True)
class FastVoice:
    id: str
    title: str
    gender: str                 # «м» / «ж»
    language: str
    engine: str                 # silero / vosk / piper
    pack: str                   # что качать: "" (встроен), "vosk", "piper:<имя>"
    license: str


CATALOG: list[FastVoice] = [
    FastVoice("aidar", "Айдар", "м", "ru", "silero", "", "CC BY-NC 4.0"),
    FastVoice("eugene", "Евгений", "м", "ru", "silero", "", "CC BY-NC 4.0"),
    FastVoice("kseniya", "Ксения", "ж", "ru", "silero", "", "CC BY-NC 4.0"),
    FastVoice("xenia", "Ксения-2", "ж", "ru", "silero", "", "CC BY-NC 4.0"),
    FastVoice("baya", "Бая", "ж", "ru", "silero", "", "CC BY-NC 4.0"),
    FastVoice("vosk:4", "Vosk 4", "м", "ru", "vosk", "vosk", "Apache-2.0"),
    FastVoice("vosk:3", "Vosk 3", "м", "ru", "vosk", "vosk", "Apache-2.0"),
    FastVoice("vosk:2", "Vosk 2", "ж", "ru", "vosk", "vosk", "Apache-2.0"),
    FastVoice("piper:en_US-joe-medium", "Joe", "м", "en", "piper",
              "piper:en_US-joe-medium", "CC0"),
    FastVoice("piper:en_US-kristin-medium", "Kristin", "ж", "en", "piper",
              "piper:en_US-kristin-medium", "общественное достояние"),
    FastVoice("piper:en_GB-cori-medium", "Cori", "ж", "en", "piper",
              "piper:en_GB-cori-medium", "общественное достояние"),
]
HIDDEN_DEFAULT = {"baya"}       # Андрей 17.10: «Бая - убрать (неприятный)»


class NoInternet(RuntimeError):
    """Нет связи с источником загрузки."""


def by_id(vid: str) -> FastVoice | None:
    return next((v for v in CATALOG if v.id == vid), None)


# --- где лежит --------------------------------------------------------------
def _data() -> Path:
    from .osutil import model_dir
    return model_dir()


def vosk_dir() -> Path:
    return Path.home() / ".cache" / "vosk" / VOSK_MODEL       # там его ищет vosk-tts


def piper_dir(name: str) -> Path:
    return _data() / "piper" / name


def piper_engine_dir() -> Path:
    return _data() / "piper_engine"


def piper_paths(name: str) -> tuple[Path, Path]:
    d = piper_dir(name)
    return d / f"{name}.onnx", d / f"{name}.onnx.json"


def installed(v: FastVoice) -> bool:
    if v.engine == "silero":
        return True
    if v.engine == "vosk":
        return (vosk_dir() / "model.onnx").is_file() and _has_module("vosk_tts")
    onnx, cfg = piper_paths(v.pack.split(":", 1)[1])
    return onnx.is_file() and cfg.is_file() and piper_engine_ready()


def size_on_disk(v: FastVoice) -> int:
    if v.engine == "silero":
        return 0
    if v.engine == "vosk":
        return _du(vosk_dir())
    return _du(piper_dir(v.pack.split(":", 1)[1]))


def download_size(v: FastVoice) -> int:
    if v.engine == "vosk":
        return VOSK_SIZE
    if v.engine == "piper":
        try:
            files = _piper_files(v.pack.split(":", 1)[1])
            return sum(f["size_bytes"] for f in files.values())
        except Exception:
            return 63_500_000
    return 0


def _du(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


# --- движок Piper: колесо с PyPI в папку данных ---------------------------------
def piper_engine_ready() -> bool:
    if _has_module("piper") and "piper_engine" not in str(_spec_origin("piper")):
        return True                                  # стоит в окружении (разработка)
    return (piper_engine_dir() / "ready.json").is_file()


def _spec_origin(name: str) -> str:
    import importlib.util
    s = importlib.util.find_spec(name)
    return (s.origin or "") if s else ""


def activate_piper() -> None:
    """Подключить скачанный движок Piper (папка данных - в путь поиска модулей)."""
    d = str(piper_engine_dir())
    if (piper_engine_dir() / "ready.json").is_file() and d not in sys.path:
        sys.path.insert(0, d)


def _wheel_tag() -> str:
    import platform
    m = platform.machine().lower()
    if sys.platform == "darwin":
        return "macosx_11_0_arm64" if m in ("arm64", "aarch64") else "macosx_10_9_x86_64"
    if sys.platform.startswith("win"):
        return "win_amd64"
    return "manylinux_2_17_aarch64" if m in ("arm64", "aarch64") else "manylinux_2_17_x86_64"


def _pypi_wheel(pkg: str, version: str, tag: str | None) -> tuple[str, str]:
    data = json.loads(_get(f"https://pypi.org/pypi/{pkg}/{version}/json").decode())
    for f in data["urls"]:
        if f["packagetype"] != "bdist_wheel":
            continue
        if tag is None and f["filename"].endswith("py3-none-any.whl"):
            return f["url"], f["digests"]["sha256"]
        if tag and tag in f["filename"]:
            return f["url"], f["digests"]["sha256"]
    raise RuntimeError(f"на PyPI нет колеса {pkg} {version} для этой системы")


def install_piper_engine(progress=None, cancel=None) -> None:
    """piper-tts (GPL-3.0) и pathvalidate - в папку данных, с проверкой sha256."""
    if piper_engine_ready():
        return
    d = piper_engine_dir()
    tmp = d.with_suffix(".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for pkg, ver, tag in (("piper-tts", PIPER_VERSION, _wheel_tag()),
                          ("pathvalidate", "3.3.1", None)):
        url, sha = _pypi_wheel(pkg, ver, tag)
        whl = tmp / url.rsplit("/", 1)[1]
        _download(url, whl, sha256=sha, progress=progress, cancel=cancel)
        with zipfile.ZipFile(whl) as z:
            z.extractall(tmp)
        whl.unlink()
    (tmp / "ready.json").write_text(json.dumps({"piper-tts": PIPER_VERSION}))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    activate_piper()


# --- загрузка голосов ------------------------------------------------------------
_PIPER_INDEX: dict | None = None


def _piper_files(name: str) -> dict:
    global _PIPER_INDEX
    if _PIPER_INDEX is None:
        _PIPER_INDEX = json.loads(_get(PIPER_REPO + "voices.json").decode())
    return {p: f for p, f in _PIPER_INDEX[name]["files"].items()
            if p.endswith((".onnx", ".onnx.json"))}


def download(v: FastVoice, progress=None, cancel=None) -> None:
    """Скачать голос (или пакет Vosk) с проверкой суммы. progress(доля, текст)."""
    if v.engine == "vosk":
        if not _has_module("vosk_tts"):
            raise RuntimeError("в этой сборке нет движка Vosk - переустановите Herald")
        z = vosk_dir().parent / (VOSK_MODEL + ".zip.part")
        z.parent.mkdir(parents=True, exist_ok=True)
        _download(VOSK_URL, z, md5=VOSK_MD5, progress=progress, cancel=cancel)
        if progress:
            progress(1.0, "Распаковываю пакет Vosk…")
        with zipfile.ZipFile(z) as zz:
            zz.extractall(vosk_dir().parent)
        z.unlink()
        return
    if v.engine == "piper":
        if not piper_engine_ready():
            if progress:
                progress(0.0, "Скачиваю движок Piper…")
            install_piper_engine(progress, cancel)
        name = v.pack.split(":", 1)[1]
        d = piper_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        for path, f in _piper_files(name).items():
            _download(PIPER_REPO + path, d / Path(path).name, md5=f["md5_digest"],
                      progress=progress, cancel=cancel)


def delete(v: FastVoice) -> None:
    if v.engine == "vosk":
        shutil.rmtree(vosk_dir(), ignore_errors=True)
    elif v.engine == "piper":
        shutil.rmtree(piper_dir(v.pack.split(":", 1)[1]), ignore_errors=True)


def sample(v: FastVoice) -> str | None:
    """Путь к образцу звучания (~10 с): Piper - официальный, без загрузки голоса."""
    if v.engine != "piper":
        return None
    name = v.pack.split(":", 1)[1]
    lang, who, q = name.split("-")
    out = _data() / "piper_samples" / f"{name}.mp3"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{PIPER_SAMPLES}{lang.split('_')[0]}/{lang}/{who}/{q}/speaker_0.mp3"
        _download(url, out)
    return str(out)


# --- сеть -------------------------------------------------------------------------
def _open(url: str, timeout: int = 30):
    import ssl
    import urllib.error
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Herald"})
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.URLError as e:
        if isinstance(e, urllib.error.HTTPError):
            raise RuntimeError(f"источник ответил ошибкой {e.code}: {url}") from e
        raise NoInternet("Нет связи с интернетом - голос не скачался. Проверьте "
                         "подключение и нажмите «Скачать» ещё раз.") from e
    except OSError as e:
        raise NoInternet("Нет связи с интернетом - голос не скачался. Проверьте "
                         "подключение и нажмите «Скачать» ещё раз.") from e


def _get(url: str) -> bytes:
    with _open(url) as r:
        return r.read()


def _download(url: str, dst: Path, md5: str | None = None, sha256: str | None = None,
              progress=None, cancel=None) -> None:
    """Скачать во временный файл, проверить сумму, затем положить на место."""
    part = dst.with_name(dst.name + ".dl")
    h_md5, h_sha = hashlib.md5(), hashlib.sha256()
    with _open(url, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            if cancel is not None and cancel():
                f.close()
                part.unlink(missing_ok=True)
                raise RuntimeError("загрузка отменена")
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h_md5.update(b)
            h_sha.update(b)
            got += len(b)
            if progress and total:
                progress(got / total, f"Скачиваю {dst.name}: {got >> 20} из {total >> 20} МБ")
    if (md5 and h_md5.hexdigest() != md5) or (sha256 and h_sha.hexdigest() != sha256):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"файл {dst.name} скачался с ошибкой (не сошлась контрольная "
                           "сумма) - попробуйте ещё раз")
    os.replace(part, dst)
