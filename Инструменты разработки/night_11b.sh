#!/bin/zsh
# Задание 11б, шаг 4 - ночная цепочка для Ященко (Герасимов - ждёт записей, < 1 ч чистых):
#  1) замок (живой процесс), 2) полное дообучение F5 (эпохи, лучшая по отложенным фразам),
#  3) голос «Игорь Ященко (F5, обучен)» в программе, 4) шипение: без чистки / DFN 15 / DFN 100
#  на абзаце - лучшее в профиль голоса, 5) длинный кусок, 6) замеры, 7) слепой набор, 8) замок.
# Журнал - Архив/11б/ночь.log.
#   zsh "Инструменты разработки/night_11b.sh"
cd "$(dirname "$0")/.."
LOG="Архив/11б/ночь.log"
T="Инструменты разработки"
W="$HOME/.cache/text2audio/train/Игорь Ященко/f5-lora-full"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

# очередь: Herald - после Interview Coach 12 и «Понятного документа» 06 (их строки - «ГОТОВО» или
# «ЖДЁТ АНДРЕЯ»); только потом замок - иначе цепочка взяла бы его в паузе между ними
Q="../Очередь проектов.md"
done_row() { grep -E "^\| $1 \|" "$Q" | grep -qE "\| (ГОТОВО|ЖДЁТ АНДРЕЯ)"; }
until done_row 2 && done_row 3; do
  say "очередь: жду Interview Coach (2) и «Понятный документ» (3)"
  sleep 300
done
zsh "$T/lock_take.sh" "11б шаг 4" >> "$LOG" 2>&1
trap 'zsh "$T/lock_free.sh" >> "$LOG" 2>&1' EXIT

say "проверка кусков (разрыв слов)"
.venv/bin/python "$T/test_chunks.py" 2>&1 | grep -a "куски до\|всё в порядке\|ПРОВАЛЕНО" >> "$LOG"

say "обучение F5, Ященко"
caffeinate -i .venv/bin/python "$T/f5_finetune_11b.py" --voice "Игорь Ященко" --epochs 8 --frames 900 \
  --lr 1e-4 --data train_clean_dfn.jsonl --out f5-lora-full 2>&1 | grep -a "эпоха\|фраз\|готово\|Error\|Traceback" >> "$LOG" || exit 1
[ -f "$W/model_v1.safetensors" ] || { say "весов нет - стоп"; exit 1; }

say "шипение: три варианта"
for x in "0:" "15: DFN15" "100: DFN100"; do
  .venv/bin/python "$T/install_f5_trained_11b.py" "Игорь Ященко" "$W" --denoise "${x%%:*}" --tag "${x#*:}" >> "$LOG"
done
for v in "" " DFN15" " DFN100"; do
  .venv/bin/python "$T/paragraph_11b.py" "Точка Г" main "Игорь Ященко (F5, обучен)$v" "Ященко - F5 обучен$v" 2>&1 | grep -a "mp3\|Error\|Traceback" >> "$LOG"
done
D="Сравнение 3.10 - лучшие варианты/Точка Г"
cp "Сравнение 3.10 - лучшие варианты/Точка А/Ященко - F5.mp3" "$D/Ященко - F5 как сейчас.mp3"
.venv/bin/python "$T/measure_11b.py" "точка Г" "Игорь Ященко" "Архив/11б/абзац.txt" "$D/Ященко - F5 как сейчас.mp3=F5 как сейчас" \
  "$D/Ященко - F5 обучен.mp3=обучен, без чистки" "$D/Ященко - F5 обучен DFN15.mp3=обучен, DFN 15" \
  "$D/Ященко - F5 обучен DFN100.mp3=обучен, DFN 100" 2>&1 | grep -a '^{' >> "$LOG"
BEST=$(.venv/bin/python - <<'EOF'
import json, numpy as np, librosa
D = "Сравнение 3.10 - лучшие варианты/Точка Г"
def pause_noise(p, sr=24000):
    a, _ = librosa.load(p, sr=sr, mono=True)
    S = np.abs(librosa.stft(a, n_fft=1024, hop_length=240)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    e = 10 * np.log10(S.sum(0) + 1e-12)
    sp, q = e > e.max() - 30, e < e.max() - 45
    b = lambda m, lo, hi: 10 * np.log10(S[(f >= lo) & (f < hi)][:, m].mean() + 1e-12)
    return float(b(q, 4000, 12000) - b(sp, 300, 3000))
rows = {r["вариант"]: r for r in (json.loads(l) for l in open("Архив/11б/замеры.jsonl", encoding="utf-8"))
        if r.get("набор") == "точка Г"}
var = {0: ("обучен, без чистки", ""), 15: ("обучен, DFN 15", " DFN15"), 100: ("обучен, DFN 100", " DFN100")}
base = rows[var[0][0]]
best, best_n = 0, pause_noise(f"{D}/Ященко - F5 обучен.mp3")
for k in (15, 100):
    r = rows.get(var[k][0])
    if not r:
        continue
    n = pause_noise(f"{D}/Ященко - F5 обучен{var[k][1]}.mp3")
    ok = r["похожесть"] >= base["похожесть"] - 0.01 and r["слов найдено, %"] >= base["слов найдено, %"] - 1.0
    if ok and n < best_n - 1.0:
        best, best_n = k, n
print(best)
EOF
)
say "шипение: выбрано denoise=$BEST (0 - без чистки)"
for v in " DFN15" " DFN100"; do .venv/bin/python "$T/install_f5_trained_11b.py" --remove "Игорь Ященко (F5, обучен)$v" >> "$LOG"; done
.venv/bin/python "$T/install_f5_trained_11b.py" "Игорь Ященко" "$W" --denoise "$BEST" >> "$LOG"

say "длинный кусок"
.venv/bin/python "$T/long_piece_39.py" f5обучен 2>&1 | grep -a "обучен)\|Error\|Traceback" >> "$LOG"
L="Сравнение 3.9 - голоса диктора на выбор/Длинный кусок"
.venv/bin/python "$T/measure_11b.py" "длинный кусок" "Игорь Ященко" "Архив/голоса диктора 3.9/длинный кусок/Глава первая (отрывок).txt" \
  "$L/Старая сборка F5/Ященко (F5).mp3=F5 как сейчас (11а)" "$L/Дообученный F5/Ященко (F5, обучен).mp3=F5 дообученный" \
  2>&1 | grep -a '^{' >> "$LOG"

say "слепой набор"
if [ "$BEST" = "0" ]; then SEL="$D/Ященко - F5 обучен.mp3"; elif [ "$BEST" = "15" ]; then SEL="$D/Ященко - F5 обучен DFN15.mp3"; else SEL="$D/Ященко - F5 обучен DFN100.mp3"; fi
.venv/bin/python "$T/blind_set_11b.py" "$SEL" "$D/Ященко - F5 как сейчас.mp3" > /dev/null 2>> "$LOG"
say "готово"
