"""Всё, что различается между macOS, Windows и Linux (по образцу Scribe 2.4).

Пути (журнал, данные), открыть файл / показать в папке, ffmpeg из установщика,
окна консоли у дочерних процессов на Windows, потоки вывода у оконной сборки
без консоли, видеокарта NVIDIA. Только стандартная библиотека.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

APP = "Herald"
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"
FROZEN = bool(getattr(sys, "frozen", False))
HAND = "pointinghand" if IS_MAC else "hand2"     # курсор-ссылка: «pointinghand» есть только в Tk для Mac
REVEAL_TEXT = "В Finder" if IS_MAC else "В папке"
HERE = "на этом Mac" if IS_MAC else "на этом компьютере"


def data_dir() -> Path:
    """Данные пользователя: настройки, голоса, корзина, поправки, кэш синтеза.
    ~/.cache/text2audio на всех системах (так было с первой версии - голоса не теряются
    при обновлении) или папка из HERALD_DATA_DIR (проверки - на своих данных, стандарт 6.10)."""
    env = os.environ.get("HERALD_DATA_DIR")
    d = Path(env).expanduser() if env else Path.home() / ".cache" / "text2audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir() -> Path:
    """Скачанные модели (Silero, F5 MLX, meter2, Qwen, словарь ударений, torch CUDA) - не
    данные пользователя: по умолчанию там же, где всегда; HERALD_MODEL_DIR - своя папка.
    От HERALD_DATA_DIR не зависит: проверка на временной папке не качает гигабайты заново."""
    env = os.environ.get("HERALD_MODEL_DIR")
    d = Path(env).expanduser() if env else Path.home() / ".cache" / "text2audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir() -> Path:
    if os.environ.get("HERALD_DATA_DIR"):
        return data_dir() / "log"
    if IS_MAC:
        return Path.home() / "Library" / "Logs" / APP
    if IS_WIN:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / APP / "Logs"
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / APP / "log"


def source_dir() -> Path:
    """Папка с ресурсами программы (assets): в сборке - распакованная _MEIPASS."""
    if FROZEN and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def open_path(path) -> None:
    path = str(path)
    if IS_WIN:
        os.startfile(path)                                  # noqa: S606 - так открывают в Windows
    elif IS_MAC:
        subprocess.run(["open", path], check=False)
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


def reveal(path) -> None:
    """Показать файл в его папке (Finder, Проводник; на Linux - открыть папку)."""
    path = Path(path)
    if IS_WIN:
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    elif IS_MAC:
        subprocess.run(["open", "-R", str(path)], check=False)
    else:
        subprocess.Popen(["xdg-open", str(path.parent)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


def setup_path() -> None:
    """ffmpeg из установщика (папка ffmpeg рядом с программой) и Homebrew на Mac."""
    extra = []
    if FROZEN:
        extra += [str(source_dir() / "ffmpeg"), str(Path(sys.executable).parent / "ffmpeg")]
    if IS_MAC:
        extra += ["/opt/homebrew/bin", "/usr/local/bin"]
    cur = os.environ.get("PATH", "").split(os.pathsep)
    for p in extra:
        if os.path.isdir(p) and p not in cur:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    if IS_WIN and FROZEN:
        _hide_console_windows()


def _hide_console_windows() -> None:
    """У оконной сборки каждый вызов ffmpeg мигал бы чёрным окном консоли."""
    if getattr(subprocess.Popen, "_herald_nowin", False):
        return
    base = subprocess.Popen

    class _Popen(base):
        _herald_nowin = True

        def __init__(self, *a, **k):
            k["creationflags"] = k.get("creationflags", 0) | 0x08000000   # CREATE_NO_WINDOW
            super().__init__(*a, **k)
    subprocess.Popen = _Popen


def fix_streams(name: str = "herald-stdout.log") -> None:
    """У оконной сборки (без консоли) sys.stdout и sys.stderr - None: любая
    печать роняла бы программу. Направляем их в журнал; вывод - всегда UTF-8."""
    if sys.stdout is None or sys.stderr is None:
        try:
            d = log_dir()
            d.mkdir(parents=True, exist_ok=True)
            f = open(d / name, "a", encoding="utf-8", buffering=1)
        except Exception:
            f = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f
    for st in (sys.stdout, sys.stderr):
        try:
            if (getattr(st, "encoding", "") or "").lower().replace("-", "") != "utf8":
                st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass                      # поток без reconfigure - оставляем как есть


def has_nvidia() -> bool:
    """Есть ли видеокарта NVIDIA с драйвером (nvidia-smi)."""
    import shutil
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(cand) if cand.exists() else None
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False
