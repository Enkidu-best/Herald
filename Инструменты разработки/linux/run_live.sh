#!/bin/bash
# Звук по мере готовности на Linux (процессор) - Herald 3.4, п. 6.
#   docker run --rm -v "$PWD":/app -v herald-cache:/root/.cache herald-linux \
#       bash "Инструменты разработки/linux/run_live.sh"
T="Инструменты разработки"
echo "=== звук потоком (заглушка движка)"
python3 "$T/test_stream.py" 2>&1 | grep -E "звук .* с,|OK|ПРОВАЛ|✗"
echo "=== Silero, фраза: первый звук"
python3 "$T/bench_live.py" --case silero --repeat 3 2>&1 | grep "первый"
echo "=== /speak, Silero: от запроса до первого звука"
python3 "$T/bench_live.py" --case server --repeat 3 2>&1 | grep "первый"
V=$(ls ~/.cache/text2audio/voices/*.txt 2>/dev/null | head -1 | xargs -I{} basename {} .txt)
if [ -n "$V" ]; then
    echo "=== клон F5 на процессоре, фраза ~10 с (голос «$V»): только время"
    python3 "$T/bench_live.py" --case phrase --voice "$V" --repeat 1 2>&1 | grep "первый"
fi
