#!/usr/bin/env bash
set -euo pipefail

PERCENTS=${PERCENTS:-"0.1"} \
PRED_LENS=${PRED_LENS:-"96"} \
VARIANTS=${VARIANTS:-"full no_ral no_val no_tal"} \
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1} \
WANDB_MODE=${WANDB_MODE:-offline} \
SUMMARY_OUTPUT_DIR=${SUMMARY_OUTPUT_DIR:-reports/weather_core_ablation_smoke} \
bash scripts/TimeVLM_weather_ablation_wandb.sh
