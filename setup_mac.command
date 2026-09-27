#!/bin/bash
# Установка «Текст в Аудио» в изолированное окружение (venv).
# Двойной клик по файлу в Finder, либо: bash setup_mac.command
set -e
cd "$(dirname "$0")"

echo "==> Проверяю ffmpeg (нужен Homebrew)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg не найден. Установи: brew install ffmpeg"; exit 1
fi

echo "==> Создаю виртуальное окружение .venv (Python 3.12)"
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv || python3.12 -m venv .venv
source .venv/bin/activate

echo "==> Обновляю pip"
python -m pip install --upgrade pip >/dev/null

echo "==> Ставлю зависимости (это займёт несколько минут)"
python -m pip install -r requirements.txt

# Убираем ПОСЛЕ всех установок: torchcodec приходит как зависимость f5-tts, и
# любая доустановка может вернуть его обратно.
echo "==> Убираю torchcodec (с torchaudio 2.6 не нужен и мешает на Homebrew-ffmpeg)"
python -m pip uninstall -y torchcodec 2>/dev/null || true

echo "==> Проверка"
python - <<'PY'
import torch, torchaudio, soundfile, f5_tts, ruaccent, faster_whisper
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__)
print("backends:", torchaudio.list_audio_backends())
try:
    import mlx.core, f5_tts_mlx, vocos_mlx
    print("MLX на месте — быстрый движок f5mlx доступен (движок по умолчанию)")
except ImportError as e:
    print("MLX недоступен (%s) — останется медленный torch-движок f5" % e.name)
print("OK — окружение готово")
PY

echo
echo "Готово. Дальше запускай так (каждый раз из этой папки):"
echo '  source .venv/bin/activate'
echo '  python cli.py "<твоя книга>.fb2" --engine f5mlx --ref "аудиокнига нужного диктора.mp3" --preview 1'
