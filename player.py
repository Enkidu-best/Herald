"""Мини-плеер для «Слушать»: системный AVAudioPlayer (macOS) через pyobjc.

Раньше «Слушать» открывал файл во внешней программе - ни паузы, ни перемотки
внутри окна. AVAudioPlayer играет mp3, m4a, flac и wav, умеет паузу, стоп и
перемотку и не требует ни своего цикла событий, ни отдельного процесса: окно
только раз в четверть секунды спрашивает у него, где он сейчас.

Пакет pyobjc-framework-AVFoundation стоит в .venv проекта. Без него (или если
файл не открылся) Player.load вернёт False, и окно откроет файл по-старому.
"""

from __future__ import annotations

import os


class Player:
    """Один звук за раз: загрузить, играть/пауза, стоп, перемотка."""

    def __init__(self) -> None:
        self._p = None
        self.path: str | None = None

    def load(self, path: str) -> bool:
        self.stop()
        self._p, self.path = None, None
        try:
            from AVFoundation import AVAudioPlayer
            from Foundation import NSURL
        except ImportError:
            return False
        if not os.path.isfile(path):
            return False
        p, err = AVAudioPlayer.alloc().initWithContentsOfURL_error_(
            NSURL.fileURLWithPath_(path), None)
        if p is None or err is not None:
            return False
        p.prepareToPlay()
        self._p, self.path = p, path
        return True

    @property
    def loaded(self) -> bool:
        return self._p is not None

    @property
    def playing(self) -> bool:
        return bool(self._p is not None and self._p.isPlaying())

    @property
    def duration(self) -> float:
        return float(self._p.duration()) if self._p is not None else 0.0

    @property
    def position(self) -> float:
        return float(self._p.currentTime()) if self._p is not None else 0.0

    def seek(self, sec: float) -> None:
        if self._p is not None:
            self._p.setCurrentTime_(max(0.0, min(float(sec), self.duration)))

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
            self._p.setCurrentTime_(0.0)


def fmt_time(sec: float) -> str:
    """«3:07», а за час - «1:05:12»."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
