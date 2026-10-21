#!/bin/zsh
# Освободить ../.heavy-lock, если он наш (Herald): остановить держащий процесс и убрать папку.
cd "$(dirname "$0")/../.."
o=$(cat .heavy-lock/owner 2>/dev/null)
case "$o" in
  Herald*) kill ${o##*pid } 2>/dev/null; rm -rf .heavy-lock; echo "$(date +%H:%M:%S) замок освобождён";;
  *) echo "замок не наш: $o";;
esac
