#!/bin/bash
# Запускатель Herald. Его вызывает «Herald.app» из корня папки программы: исполняемый файл пакета - программа на C из
# общего сборщика «_общие модули/app» (стандарт 6.2; до 25.10 исполняемым файлом был bash-скрипт - macOS могла молча не
# дать ему доступ к «Рабочему столу»). Скрипт правится без пересборки. Открывает окно Herald (Tk).
#   HERALD_LAUNCH_CHECK=<порт> - вместо окна служба /speak на этом порту: проверка сборки без окна на экране
#   (правило 10), версия - в /health. HERALD_DATA_DIR - своя папка данных.
export PYTHONHASHSEED=0          # должен быть числом, иначе падают дочерние процессы
export PYTORCH_ENABLE_MPS_FALLBACK=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
export PATH="$ROOT/.venv/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
LOG="$HOME/Library/Logs/Herald.log"
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date '+%d.%m.%Y %H:%M:%S') $*" >>"$LOG"; }

if [ ! -d "$ROOT" ] || ! cd "$ROOT" 2>/dev/null || ! head -c1 "$ROOT/tts_gui.py" >/dev/null 2>&1; then
  say "нет доступа к папке программы (защита macOS) или папку перенесли: $ROOT"
  osascript -e 'display dialog "macOS не дал Herald доступ к папке программы (обычно «Рабочий стол»).\n\nСистемные настройки → Конфиденциальность и безопасность → Файлы и папки → Herald → «Папка „Рабочий стол“». Затем запустите программу ещё раз." with title "Herald" buttons {"Открыть настройки", "Закрыть"} default button "Открыть настройки" with icon caution' \
    -e 'if button returned of result is "Открыть настройки" then do shell script "open \"x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders\""' >/dev/null 2>&1
  exit 1
fi
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  say "нет окружения .venv"
  osascript -e 'display alert "Herald" message "Окружение не найдено. Запустите setup_mac.command в папке программы." as critical' >/dev/null 2>&1
  exit 1
fi
if [ -n "${HERALD_LAUNCH_CHECK:-}" ]; then
  say "проверка сборки: служба /speak на порту $HERALD_LAUNCH_CHECK, окна нет"
  set -- cli.py --serve --port "$HERALD_LAUNCH_CHECK"
else
  say "запуск окна"
  set -- tts_gui.py
fi
# arch -arm64 обязателен: запущенный из Finder пакет может уехать в ветку x86_64 (Rosetta), а numpy / mlx собраны под arm64
if [ "$(uname -m)" = "arm64" ] && command -v arch >/dev/null; then
  exec arch -arm64 "$PY" "$@" >>"$LOG" 2>&1
fi
exec "$PY" "$@" >>"$LOG" 2>&1
