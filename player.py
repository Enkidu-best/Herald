"""Мини-плеер для «Слушать»: играть/пауза, стоп, перемотка, время.

Два способа, один интерфейс:
  * macOS - системный AVAudioPlayer через pyobjc (pyobjc-framework-AVFoundation):
    играет mp3, m4a, flac и wav сам, окно лишь раз в четверть секунды спрашивает,
    где он сейчас;
  * Windows и Linux (и Mac без pyobjc) - ffmpeg раскодирует файл с нужной
    секунды, звук идёт через sounddevice (PortAudio). Перемотка = ffmpeg заново
    с новой секунды; пауза - остановка потока вывода.
Если не вышло ни так, ни так, Player.load вернёт False, и окно откроет файл
внешней программой, как раньше.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading


class _AVPlayer:
    """macOS: AVAudioPlayer."""

    def __init__(self, path: str):
        from AVFoundation import AVAudioPlayer
        from Foundation import NSURL
        p, err = AVAudioPlayer.alloc().initWithContentsOfURL_error_(
            NSURL.fileURLWithPath_(path), None)
        if p is None or err is not None:
            raise RuntimeError(f"AVAudioPlayer не открыл файл: {err}")
        p.prepareToPlay()
        self._p = p

    playing = property(lambda self: bool(self._p.isPlaying()))
    duration = property(lambda self: float(self._p.duration()))
    position = property(lambda self: float(self._p.currentTime()))

    def seek(self, sec):
        self._p.setCurrentTime_(max(0.0, min(float(sec), self.duration)))

    def play(self):
        self._p.play()

    def pause(self):
        self._p.pause()

    def stop(self):
        self._p.stop()
        self._p.setCurrentTime_(0.0)


class _FFPlayer:
    """Windows, Linux: ffmpeg -> PCM -> sounddevice."""
    SR = 48000

    def __init__(self, path: str):
        import sounddevice  # noqa: F401 - нет пакета/звуковой карты - пусть упадёт здесь
        self.path = path
        self._dur = self._probe(path)
        self._offset = 0.0           # с какой секунды запущен ffmpeg
        self._frames = 0             # сколько кадров уже отдано звуковой карте
        self._proc = None
        self._stream = None
        self._lock = threading.Lock()
        self._done = False

    @staticmethod
    def _probe(path: str) -> float:
        exe = shutil.which("ffprobe") or "ffprobe"
        out = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", path], capture_output=True, text=True).stdout
        try:
            return float(out.strip())
        except ValueError:
            raise RuntimeError("ffprobe не узнал длительность файла")

    def _open(self, start: float):
        import sounddevice as sd
        self._close()
        exe = shutil.which("ffmpeg") or "ffmpeg"
        self._proc = subprocess.Popen(
            [exe, "-v", "error", "-ss", f"{start:.3f}", "-i", self.path, "-f", "s16le",
             "-ac", "1", "-ar", str(self.SR), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._offset, self._frames, self._done = start, 0, False

        def cb(outdata, frames, _time, _status):
            need = frames * 2
            data = self._proc.stdout.read(need) if self._proc else b""
            if len(data) < need:
                self._done = True
                data += b"\0" * (need - len(data))
                outdata[:] = data
                raise sd.CallbackStop
            outdata[:] = data
            self._frames += frames
        self._stream = sd.RawOutputStream(samplerate=self.SR, channels=1, dtype="int16",
                                          callback=cb, blocksize=2048)

    def _close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass                 # поток уже закрыт
            self._stream = None
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass                 # ffmpeg уже завершился
            self._proc = None

    @property
    def playing(self) -> bool:
        return bool(self._stream is not None and self._stream.active and not self._done)

    @property
    def duration(self) -> float:
        return self._dur

    @property
    def position(self) -> float:
        if self._done:
            return 0.0 if self._stream is None else self._dur
        return min(self._dur, self._offset + self._frames / self.SR)

    def seek(self, sec):
        sec = max(0.0, min(float(sec), self._dur))
        was = self.playing
        self._open(sec)
        if was:
            self._stream.start()

    def play(self):
        if self._stream is None or self._done:
            self._open(0.0 if self._done else self._offset)
        self._stream.start()

    def pause(self):
        if self._stream is not None and self._stream.active:
            pos = self.position
            self._stream.stop()
            self._offset, self._frames = pos, 0
            self._open(pos)          # продолжение - с той же секунды

    def stop(self):
        self._close()
        self._offset, self._frames, self._done = 0.0, 0, False


class Player:
    """Один звук за раз: загрузить, играть/пауза, стоп, перемотка."""

    def __init__(self) -> None:
        self._p = None
        self.path: str | None = None

    def load(self, path: str) -> bool:
        self.stop()
        self._p, self.path = None, None
        if not os.path.isfile(path):
            return False
        kinds = ([_AVPlayer] if sys.platform == "darwin" else []) + [_FFPlayer]
        for kind in kinds:
            try:
                self._p = kind(path)
                self.path = path
                return True
            except Exception:
                continue             # следующий способ
        return False

    @property
    def loaded(self) -> bool:
        return self._p is not None

    @property
    def playing(self) -> bool:
        return bool(self._p is not None and self._p.playing)

    @property
    def duration(self) -> float:
        return float(self._p.duration) if self._p is not None else 0.0

    @property
    def position(self) -> float:
        return float(self._p.position) if self._p is not None else 0.0

    def seek(self, sec: float) -> None:
        if self._p is not None:
            self._p.seek(sec)

    def play(self) -> None:
        if self._p is not None:
            self._p.play()

    def pause(self) -> None:
        if self._p is not None:
            self._p.pause()

    def toggle(self) -> bool:
        """Играть/пауза одной кнопкой. Возвращает: играет ли теперь."""
        if self.playing:
            self.pause()
        else:
            self.play()
        return self.playing

    def stop(self) -> None:
        """Стоп: остановить и вернуться в начало."""
        if self._p is not None:
            self._p.stop()


def fmt_time(sec: float) -> str:
    """«3:07», а за час - «1:05:12»."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
