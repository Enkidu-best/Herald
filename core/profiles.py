"""Профили озвучки: какой движок и с какими настройками.

Выбраны слепым прослушиванием «Сравнение голосов 3.0» (14.10): на
повествовании - главном в книге - лучшим оказался F5 Misha accent_tune на 16
шагах (вариант B16), вторым - ESpeech-TTS-1 RL-V2 на 16 шагах (C16), у него
свободная лицензия Apache-2.0. Прежние 7 шагов остались «черновиком»: вдвое
быстрее, но с шумом и металлом. Qwen3-TTS и Chatterbox в программу не вошли.

rtf - сколько секунд счёта уходит на секунду звука на Apple M4 (замер на
повествовании и диалоге; на отрывке главы «Жажды власти» 4 мин: основной 2,28,
ESpeech 1,83). По нему окно честно говорит, сколько ждать.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    key: str
    title: str
    engine: str
    kwargs: dict = field(default_factory=dict)
    rtf: float = 1.0
    hint: str = ""
    clone: bool = True


PROFILES: dict[str, Profile] = {p.key: p for p in (
    Profile("main", "Основной (рекомендуется)", "f5mlx",
            {"steps": 16, "schedule": "epss"}, rtf=2.3,
            hint="Лучший по слуху голос. Час звука - около 2 ч 20 мин счёта: книгу "
                 "удобно ставить на ночь, «Продолжить озвучку» и запрет сна есть."),
    Profile("espeech", "ESpeech (свободная лицензия)", "f5mlx",
            {"steps": 16, "schedule": "epss", "variant": "espeech"}, rtf=1.85,
            hint="Почти так же хорош и чуть быстрее (час звука - около 1 ч 50 мин), "
                 "лицензия Apache-2.0 - можно и для продукта. Первый запуск скачивает 2,7 ГБ."),
    Profile("draft", "Быстрый черновик (7 шагов)", "f5mlx",
            {"steps": 7, "schedule": "epss"}, rtf=1.0,
            hint="Вдвое быстрее основного, но с шумом и металлическим призвуком. "
                 "Для проверки, не для прослушивания."),
    Profile("silero", "Готовые голоса Silero", "silero", {}, rtf=0.03, clone=False,
            hint="Не клон: готовые голоса (Айдар, Бая, Ксения…). Очень быстро - час "
                 "звука за пару минут. Лицензия некоммерческая."),
)}
DEFAULT_PROFILE = "main"


def get(key: str | None) -> Profile:
    return PROFILES.get(key or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
