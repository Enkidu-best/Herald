"""Herald - ядро приложения: чтение книг, подготовка текста, синтез, сборка."""

APP_NAME = "Herald"
APP_TAGLINE = "книги, прочитанные вслух"


def _version() -> str:
    """Номер версии - из одного места, файла core/VERSION (задание 13 Г6): окно, Справка,
    /health и установщик Windows берут его отсюда."""
    import os
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION"), encoding="utf-8") as f:
            return f.read().strip() or "?"
    except OSError:
        return "?"


VERSION = _version()
