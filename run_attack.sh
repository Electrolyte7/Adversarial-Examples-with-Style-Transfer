#!/usr/bin/env bash
set -euo pipefail

CONTENT_DIR="data/content"
STYLE_PATH="data/style/style.jpg"
OUTPUT_TAG="style"
OUTPUT_DIR="outputs_${OUTPUT_TAG}"

ITERS=1000
LR=0.05
LAMBDA_ADV=25
ALPHA=30000
BETA=0.2

DEVICE="cuda"
GPU_ID="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --content_dir) CONTENT_DIR="$2"; shift 2;;
    --style_path)  STYLE_PATH="$2";  shift 2;;
    --output_dir)  OUTPUT_DIR="$2";  shift 2;;
    --iters)       ITERS="$2";       shift 2;;
    --lr)          LR="$2";          shift 2;;
    --lambda_adv)  LAMBDA_ADV="$2";  shift 2;;
    --alpha)       ALPHA="$2";       shift 2;;
    --beta)        BETA="$2";        shift 2;;
    --device)      DEVICE="$2";      shift 2;;
    --gpu)         GPU_ID="$2";      shift 2;;
    *) shift;;
  esac
done

mkdir -p "${OUTPUT_DIR}/adv_images" "${OUTPUT_DIR}/preds" "${OUTPUT_DIR}/logs"

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

python -m src.attack \
  --content_dir "${CONTENT_DIR}" \
  --style_path "${STYLE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --iters "${ITERS}" \
  --lr "${LR}" \
  --lambda_adv "${LAMBDA_ADV}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --device "${DEVICE}"
