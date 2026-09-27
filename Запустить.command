#!/bin/bash
# Двойной щелчок по этому файлу в Finder запускает приложение.
# (Если Finder не даёт запустить: правый клик -> Открыть -> Открыть.)
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Окружение не найдено. Сначала запусти setup_mac.command"
  echo "Нажми Enter, чтобы закрыть."; read; exit 1
fi

source .venv/bin/activate
# PYTHONHASHSEED должен быть числом: иначе падают вспомогательные процессы
export PYTHONHASHSEED=0
export PYTORCH_ENABLE_MPS_FALLBACK=1

python tts_gui.py
code=$?
if [ $code -ne 0 ]; then
  echo
  echo "Программа завершилась с ошибкой (код $code)."
  echo "Нажми Enter, чтобы закрыть."; read
fi
