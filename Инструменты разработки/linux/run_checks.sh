#!/bin/bash
# Проверки Herald в контейнере Linux (процессор) - Herald 3.2.
#   docker run --rm -v "$PWD":/app -v herald-cache:/root/.cache herald-linux \
#       bash "Инструменты разработки/linux/run_checks.sh"
T="Инструменты разработки"
echo "=== чем считается";      python3 cli.py --engine-info
echo "=== сверка слов";        python3 "$T/test_words.py" | tail -1
echo "=== паузы (без модели)"; python3 "$T/test_pauses.py" --fast | tail -1
echo "=== ударения";           python3 "$T/test_stress.py" | tail -6
echo "=== нормализация";      python3 "$T/test_normalize.py"
echo "=== пробник Silero -> образец -> клон F5"; python3 "$T/smoke_pc.py" | grep -E "✓|✗|услышано|OK|ПРОВАЛ"
echo "=== пробник клоном на 1 минуту"
V=$(ls ~/.cache/text2audio/voices/*.txt | head -1 | xargs -I{} basename {} .txt)
start=$(date +%s)
python3 cli.py "Примеры текстов/Пример - проба пера.txt" --voice "$V" --preview 1 --out /tmp/prev 2>&1 | grep -E "Пробник|Длительность"
echo "    счёт: $(( $(date +%s) - start )) с"
echo "=== окончания (1 зерно)"; python3 "$T/test_endings.py" --seeds 1 2>&1 | grep -E "зерно|Сырые|Прежний|OK|ПРОВАЛ|  -"
