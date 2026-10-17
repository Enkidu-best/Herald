"""Куда идёт звук потоком (3.8): текущее устройство системы, на лету.

Андрей: «звук идёт не в наушники». PortAudio (sounddevice) запоминает список
устройств при первом обращении и открывает поток на конкретном устройстве:
наушники, подключённые после запуска программы, он не видит - ни для идущего
звука, ни для нового. Поэтому:
  * «Как в системе» (по умолчанию): на Mac устройство системы читается из
    CoreAudio; сменилось - PortAudio перечитывает устройства, и поток
    переоткрывается на новом месте с той же точки (сэмплы ждут в очереди
    _Sink, теряется один блок ~20 мс);
  * Windows: PortAudio по умолчанию идёт через MME «Microsoft Sound Mapper» -
    его Windows сама переводит на новое устройство; Linux: PulseAudio /
    PipeWire переносят поток сами. Вживую не проверено - на Mac проверено
    (переключение устройства системы во время чтения, scenario_audio_out.py);
  * своё устройство (Настройки → «Устройство вывода») - по имени.
"""

from __future__ import annotations

import ctypes
import sys
import threading

SYSTEM = "system"
_setting = SYSTEM                  # окно задаёт из настроек
_pa_default: int | None = None     # устройство системы, когда PortAudio читал список
_open_streams = 0
_lock = threading.RLock()


def set_output(name: str | None) -> None:
    global _setting
    _setting = name or SYSTEM


def output_setting() -> str:
    return _setting


# --- Mac: устройство системы из CoreAudio ----------------------------------------
class _Addr(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]


def _cc(s: str) -> int:
    return int.from_bytes(s.encode("ascii"), "big")


_ca = None


def _coreaudio():
    global _ca
    if _ca is None and sys.platform == "darwin":
        _ca = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        for fn in ("AudioObjectGetPropertyData", "AudioObjectSetPropertyData"):
            getattr(_ca, fn).restype = ctypes.c_int32
    return _ca


def system_default() -> int | None:
    """Номер устройства вывода системы (Mac); None - не знаем (другие системы)."""
    ca = _coreaudio()
    if ca is None:
        return None
    a = _Addr(_cc("dOut"), _cc("glob"), 0)
    dev = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    if ca.AudioObjectGetPropertyData(1, ctypes.byref(a), 0, None, ctypes.byref(size),
                                     ctypes.byref(dev)) != 0:
        return None
    return int(dev.value)


def set_system_default(dev_id: int) -> bool:
    """Сменить устройство вывода системы (только для проверок на Mac)."""
    ca = _coreaudio()
    if ca is None:
        return False
    a = _Addr(_cc("dOut"), _cc("glob"), 0)
    dev = ctypes.c_uint32(dev_id)
    return ca.AudioObjectSetPropertyData(1, ctypes.byref(a), 0, None, 4, ctypes.byref(dev)) == 0


def running_somewhere(dev_id: int) -> bool:
    """Играет ли что-то на этом устройстве (Mac, для проверок)."""
    ca = _coreaudio()
    a = _Addr(_cc("gone"), _cc("glob"), 0)
    v = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    ca.AudioObjectGetPropertyData(dev_id, ctypes.byref(a), 0, None, ctypes.byref(size),
                                  ctypes.byref(v))
    return bool(v.value)


def all_output_ids() -> list[int]:
    """Все устройства CoreAudio (Mac, для проверок)."""
    ca = _coreaudio()
    a = _Addr(_cc("dev#"), _cc("glob"), 0)
    size = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(1, ctypes.byref(a), 0, None, ctypes.byref(size))
    n = size.value // 4
    arr = (ctypes.c_uint32 * n)()
    ca.AudioObjectGetPropertyData(1, ctypes.byref(a), 0, None, ctypes.byref(size), arr)
    return [int(x) for x in arr]


def has_output(dev_id: int) -> bool:
    ca = _coreaudio()
    a = _Addr(_cc("stm#"), _cc("outp"), 0)
    size = ctypes.c_uint32(0)
    ca.AudioObjectGetPropertyDataSize(dev_id, ctypes.byref(a), 0, None, ctypes.byref(size))
    return size.value > 0


# --- PortAudio -------------------------------------------------------------------
def _refresh_portaudio(force: bool = False) -> None:
    """Перечитать устройства, если устройство системы сменилось (и наших
    потоков сейчас нет - иначе PortAudio их оборвёт)."""
    global _pa_default
    import sounddevice as sd
    cur = system_default()
    with _lock:
        if _pa_default is None:
            _pa_default = cur
            return
        # force - поток только что закрыт нами (переход на другое устройство);
        # на Mac PortAudio больше никто не держит (мини-плеер - AVFoundation)
        if force or (cur != _pa_default and _open_streams == 0):
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
            _pa_default = cur


def outputs() -> list[str]:
    """Имена устройств вывода (для Настроек)."""
    import sounddevice as sd
    _refresh_portaudio()
    try:
        return [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
    except Exception:
        return []


def device_for_stream():
    """Что передать sounddevice: None - устройство системы, или номер по имени."""
    import sounddevice as sd
    _refresh_portaudio()
    if _setting == SYSTEM:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == _setting and d["max_output_channels"] > 0:
            return i
    return None                      # устройства нет (отключили) - как в системе


def stream_opened() -> None:
    global _open_streams
    with _lock:
        _open_streams += 1


def stream_closed() -> None:
    global _open_streams
    with _lock:
        _open_streams = max(0, _open_streams - 1)
