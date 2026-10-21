#!/bin/zsh
# Задание 11а, часть Д - ночная цепочка: ждёт ../.heavy-lock, держит его живым процессом
# python (чтобы замок не сочли брошенным), обучает голоса на чистых данных, ставит их
# в программу, подгоняет темп под живого чтеца, озвучивает длинный кусок и меряет.
# Журнал - Архив/голоса диктора 3.9/обученный голос/ночь.log.
#   zsh "Инструменты разработки/night_11a.sh"
cd "$(dirname "$0")/.."
LOCK="../.heavy-lock"
OUT="Архив/голоса диктора 3.9/обученный голос"
LOG="$OUT/ночь.log"
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

until mkdir "$LOCK" 2>/dev/null; do
  say "замок занят: $(cat $LOCK/owner 2>/dev/null) - жду 5 мин"
  sleep 300
done
.venv/bin/python -c "import time; time.sleep(6*3600)" &
HOLD=$!
echo "Herald (11а Д, ночь) $(date) pid $HOLD" > "$LOCK/owner"
say "замок взят, держит pid $HOLD"
trap 'kill $HOLD 2>/dev/null; rm -rf "$LOCK"; say "замок освобождён"' EXIT

for v in "Игорь Ященко" "Вячеслав Герасимов"; do
  rm -rf ~/.cache/text2audio/train/"$v"/lora-clean
  caffeinate -i .venv-train/bin/python "Инструменты разработки/train_lora_11a.py" --voice "$v" --epochs 3 \
    --lr 5e-5 --data train_clean.jsonl --out lora-clean 2>&1 | grep -a -v "Warning\|warn" >> "$LOG" || exit 1
  .venv-mlxaudio/bin/python "Инструменты разработки/lora_to_mlx_11a.py" --voice "$v" --lora lora-clean \
    2>&1 | grep -a "эпох\|готово\|Error" >> "$LOG"
done
.venv/bin/python "Инструменты разработки/install_trained_11a.py" >> "$LOG" 2>&1
rm -f ~/.cache/text2audio/voices/"Игорь Ященко (обучен).fx.json" ~/.cache/text2audio/voices/"Вячеслав Герасимов (обучен).fx.json"

say "слова и темп (без поправки темпа)"
.venv/bin/python "Инструменты разработки/trained_words_11a.py" "Игорь Ященко (обучен)|0.8|0.85|30" \
  "Вячеслав Герасимов (обучен)|0.8|0.85|30" 2>&1 | grep -a '^{' >> "$LOG"
# темп живых чтецов на их записях (тем же способом): 5,82 / 5,64 слога в секунду речи
.venv/bin/python - <<'EOF' >> "$LOG" 2>&1
import json, os
target = {"Игорь Ященко": 5.82, "Вячеслав Герасимов": 5.64}
rows = [json.loads(l) for l in open("Архив/голоса диктора 3.9/обученный голос/слова.jsonl", encoding="utf-8")]
vd = os.path.expanduser("~/.cache/text2audio/voices")
for reader, t in target.items():
    r = [x for x in rows if x["вариант"].startswith(f"{reader} (обучен) t0.8")][-1]
    sp = round(min(1.0, max(0.8, t / r["слогов в секунду речи"])), 3)
    json.dump({"speed": sp, "pitch": 0.0, "bass": 0.0, "brightness": 0.0},
              open(f"{vd}/{reader} (обучен).fx.json", "w", encoding="utf-8"), indent=2)
    print(f"{reader}: темп {r['слогов в секунду речи']} -> цель {t}, speed {sp}; слов {r['%']} %")
EOF

say "длинный кусок"
.venv/bin/python "Инструменты разработки/long_piece_39.py" обучен 2>&1 | grep -a "обучен)\|Error" >> "$LOG"
say "замер длинного куска"
.venv/bin/python "Инструменты разработки/lora_eval_11a.py" 2>&1 | grep -a "Ященко\|Герасимов\|Error" >> "$LOG"
say "ударения: 50 трудных слов"
for v in "Игорь Ященко (обучен)" "Вячеслав Герасимов (обучен)"; do
  VOICE="$v" RUN="$v" .venv/bin/python "Инструменты разработки/stress50_39.py" 2>&1 | grep -a "верно\|Error" >> "$LOG"
done
say "готово"
