#!/bin/zsh
# Взять ../.heavy-lock (ждать, пока свободен) и держать его живым процессом python:
# замок без процесса старше 3 ч считается брошенным (правило очереди) - 21.10 так и сняли.
#   zsh "Инструменты разработки/lock_take.sh" "<что делаю>"
cd "$(dirname "$0")/../.."
until mkdir .heavy-lock 2>/dev/null; do
  echo "$(date +%H:%M:%S) замок занят: $(cat .heavy-lock/owner 2>/dev/null) - жду 1 мин"
  sleep 60
done
nohup "Текст в Аудио/.venv/bin/python" -c "import time; time.sleep(12*3600)" >/dev/null 2>&1 &
echo "Herald ($1) $(date) pid $!" > .heavy-lock/owner
echo "$(date +%H:%M:%S) замок взят: $(cat .heavy-lock/owner)"
