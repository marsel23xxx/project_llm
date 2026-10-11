#!/usr/bin/env bash
# Unduh model LLM lokal (~2 GB) ke folder models/
set -euo pipefail

MODEL_FILE="${LOCAL_LLM_FILE:-qwen2.5-coder-3b-instruct-q4_k_m.gguf}"
URL="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/${MODEL_FILE}"
DEST="$(dirname "$0")/../models/${MODEL_FILE}"

mkdir -p "$(dirname "$DEST")"
if [[ -f "$DEST" ]]; then
  echo "Model sudah ada: $DEST"
  exit 0
fi

echo "Mengunduh $MODEL_FILE ..."
curl -L --fail --progress-bar -o "$DEST.part" "$URL"
mv "$DEST.part" "$DEST"
echo "Selesai: $DEST"
