"""Ручная подстройка голоса: высота, бас, яркость.

Что здесь можно, а что нет. Характер речи — интонация, манера, паузы, акцент —
«запечён» в образце: модель копирует его целиком, и никакой ручкой это не
меняется. Зато после синтеза звук можно подправить как на микшерном пульте, и
три вещи слышны сразу:

    темп     — быстрее или медленнее читает (задаётся при синтезе);
    высота   — тот же диктор, но голос ниже или выше (сдвиг в полутонах);
    бас      — «телесность», насколько голос грудной;
    яркость  — воздух и чёткость согласных.

Темп тут же, а не в главном окне: это свойство КОНКРЕТНОГО диктора, и
подбирается он на слух вместе с тембром. Работает он иначе, чем остальные
три — не постобработкой, а прямо при синтезе (см. target_bps в f5_mlx).

Настройки хранятся рядом с голосом, файлом «<имя>.fx.json», и применяются при
каждой озвучке этим голосом.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

SR = 48000            # частота на выходе полировки (см. assemble.HD_FILTER)


@dataclass
class VoiceFX:
    """Настройки голоса. Значения по умолчанию — звучание как есть."""
    speed: float = 1.0        # множитель темпа, 0.8…1.25
    pitch: float = 0.0        # полутона, -4…+4
    bass: float = 0.0         # дБ на 110 Гц, -6…+6
    brightness: float = 0.0   # дБ полкой с 7 кГц, -6…+6

    def is_neutral(self) -> bool:
        return (self.speed == 1.0 and not (self.pitch or self.bass
                                           or self.brightness))

    def filter_chain(self) -> str:
        """Цепочка фильтров ffmpeg. Пустая строка — если менять нечего."""
        parts: list[str] = []
        if self.pitch:
            # Сдвиг высоты «на бедного»: ускоряем проигрывание (вместе с
            # высотой), а темп возвращаем обратно. Фильтр rubberband сделал бы
            # это чище, но в сборке ffmpeg из Homebrew его обычно нет, а этот
            # приём работает везде и на ±4 полутона звучит прилично.
            ratio = 2 ** (self.pitch / 12)
            parts.append(f"asetrate={SR}*{ratio:.6f}")
            parts.append(f"aresample={SR}")
            parts.append(f"atempo={1 / ratio:.6f}")
        if self.bass:
            parts.append(f"equalizer=f=110:t=q:w=1.0:g={self.bass:.2f}")
        if self.brightness:
            parts.append(f"equalizer=f=7000:t=h:w=2500:g={self.brightness:.2f}")
        return ",".join(parts)


def fx_path(voices_dir: str, voice_id: str) -> str:
    return os.path.join(voices_dir, f"{voice_id}.fx.json")


def load(voices_dir: str, voice_id: str) -> VoiceFX:
    try:
        with open(fx_path(voices_dir, voice_id), encoding="utf-8") as f:
            data = json.load(f)
        return VoiceFX(**{k: float(v) for k, v in data.items()
                          if k in VoiceFX.__dataclass_fields__})
    except Exception:
        return VoiceFX()


def save(voices_dir: str, voice_id: str, fx: VoiceFX) -> None:
    path = fx_path(voices_dir, voice_id)
    if fx.is_neutral():
        if os.path.exists(path):
            os.remove(path)          # нечего хранить — не мусорим файлами
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(fx), f, ensure_ascii=False, indent=2)
