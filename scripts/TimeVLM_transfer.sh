#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

model_name=${MODEL_NAME:-TimeVLM}
vlm_type=${VLM_TYPE:-vilt}
gpu=${GPU:-0}
image_size=${IMAGE_SIZE:-56}
norm_const=${NORM_CONST:-0.4}
three_channel_image=${THREE_CHANNEL_IMAGE:-True}
finetune_vlm=${FINETUNE_VLM:-False}
batch_size=${BATCH_SIZE:-32}
num_workers=${NUM_WORKERS:-32}
learning_rate=${LEARNING_RATE:-0.001}
seq_len=${SEQ_LEN:-512}
train_epochs=${TRAIN_EPOCHS:-15}
seed=${SEED:-2021}
log_to_console=${LOG_TO_CONSOLE:-True}

use_wandb=${USE_WANDB:-True}
wandb_project=${WANDB_PROJECT:-TimeVLM}
wandb_entity=${WANDB_ENTITY:-}
wandb_mode=${WANDB_MODE:-online}
wandb_group=${WANDB_GROUP:-zero-shot-transfer}
wandb_tags=${WANDB_TAGS:-zero-shot,transfer}

mkdir -p logs/zero_shot

# Function to run zero-shot forecast experiment
run_zero_shot_experiment() {
    local source_data=$1
    local target_data=$2
    local n_vars=$3
    local pred_len=$4
    local d_model=$5
    local use_mem_gate=$6
    local periodicity=$7

    local source_path="./dataset/${source_data}.csv"
    local target_path="./dataset/${target_data}.csv"
    local run_name="zero_shot_${source_data}_to_${target_data}_sl${seq_len}_pl${pred_len}_seed${seed}_${vlm_type}"
    local log_file="logs/zero_shot/${run_name}.log"

    if [ ! -f "$source_path" ]; then
        echo "Missing ${source_path}. Download the ETT datasets under ./dataset before running."
        exit 1
    fi
    if [ ! -f "$target_path" ]; then
        echo "Missing ${target_path}. Download the ETT datasets under ./dataset before running."
        exit 1
    fi

    echo "Running ${run_name}: source=${source_data}, target=${target_data}, seq_len=${seq_len}, pred_len=${pred_len}"

    cmd=(
      python -u run.py
      --task_name "zero_shot_forecast"
      --is_training 1
      --root_path ./dataset/
      --data_path "${source_data}.csv"
      --model_id "${source_data}_${target_data}_${seq_len}_${pred_len}"
      --model "$model_name"
      --data "$source_data"
      --features M
      --seq_len "$seq_len"
      --label_len 48
      --pred_len "$pred_len"
      --e_layers 2
      --d_layers 1
      --factor 3
      --enc_in "$n_vars"
      --dec_in "$n_vars"
      --c_out "$n_vars"
      --des "Exp"
      --itr 1
      --gpu "$gpu"
      --use_amp
      --train_epochs "$train_epochs"
      --d_model "$d_model"
      --image_size "$image_size"
      --norm_const "$norm_const"
      --periodicity "$periodicity"
      --three_channel_image "$three_channel_image"
      --finetune_vlm "$finetune_vlm"
      --batch_size "$batch_size"
      --learning_rate "$learning_rate"
      --num_workers "$num_workers"
      --vlm_type "$vlm_type"
      --use_mem_gate "$use_mem_gate"
      --target_data "$target_data"
      --target_root_path ./dataset/
      --target_data_path "${target_data}.csv"
      --seed "$seed"
      --use_wandb "$use_wandb"
      --wandb_project "$wandb_project"
      --wandb_entity "$wandb_entity"
      --wandb_group "$wandb_group"
      --wandb_run_name "$run_name"
      --wandb_tags "$wandb_tags,${source_data}-to-${target_data},pred-len-${pred_len}"
      --wandb_mode "$wandb_mode"
    )

    if [ "$log_to_console" = "True" ] || [ "$log_to_console" = "true" ] || [ "$log_to_console" = "1" ]; then
        "${cmd[@]}" 2>&1 | tee "$log_file"
    else
        "${cmd[@]}" > "$log_file" 2>&1
    fi
}

# ETTh1 -> ETTh2
run_zero_shot_experiment "ETTh1" "ETTh2" 7 96 64 True 24
run_zero_shot_experiment "ETTh1" "ETTh2" 7 192 64 True 24
run_zero_shot_experiment "ETTh1" "ETTh2" 7 336 64 True 24
run_zero_shot_experiment "ETTh1" "ETTh2" 7 720 64 True 24

# ETTh1 -> ETTm2
run_zero_shot_experiment "ETTh1" "ETTm2" 7 96 64 True 24
run_zero_shot_experiment "ETTh1" "ETTm2" 7 192 64 True 24
run_zero_shot_experiment "ETTh1" "ETTm2" 7 336 64 True 24
run_zero_shot_experiment "ETTh1" "ETTm2" 7 720 64 True 24

# ETTh2 -> ETTh1
run_zero_shot_experiment "ETTh2" "ETTh1" 7 96 64 True 24
run_zero_shot_experiment "ETTh2" "ETTh1" 7 192 64 True 24
run_zero_shot_experiment "ETTh2" "ETTh1" 7 336 64 True 24
run_zero_shot_experiment "ETTh2" "ETTh1" 7 720 64 True 24

# ETTh2 -> ETTm2
run_zero_shot_experiment "ETTh2" "ETTm2" 7 96 64 True 24
run_zero_shot_experiment "ETTh2" "ETTm2" 7 192 64 True 24
run_zero_shot_experiment "ETTh2" "ETTm2" 7 336 64 True 24
run_zero_shot_experiment "ETTh2" "ETTm2" 7 720 64 True 24

# ETTm1 -> ETTh2
run_zero_shot_experiment "ETTm1" "ETTh2" 7 96 64 True 24
run_zero_shot_experiment "ETTm1" "ETTh2" 7 192 64 True 24
run_zero_shot_experiment "ETTm1" "ETTh2" 7 336 64 True 24
run_zero_shot_experiment "ETTm1" "ETTh2" 7 720 64 True 24

# ETTm1 -> ETTm2
run_zero_shot_experiment "ETTm1" "ETTm2" 7 96 64 True 24
run_zero_shot_experiment "ETTm1" "ETTm2" 7 192 64 True 24
run_zero_shot_experiment "ETTm1" "ETTm2" 7 336 64 True 24
run_zero_shot_experiment "ETTm1" "ETTm2" 7 720 64 True 24

# ETTm2 -> ETTh2
run_zero_shot_experiment "ETTm2" "ETTh2" 7 96 64 True 24
run_zero_shot_experiment "ETTm2" "ETTh2" 7 192 64 True 24
run_zero_shot_experiment "ETTm2" "ETTh2" 7 336 64 True 24
run_zero_shot_experiment "ETTm2" "ETTh2" 7 720 64 True 24

# ETTm2 -> ETTm1
run_zero_shot_experiment "ETTm2" "ETTm1" 7 96 64 True 24
run_zero_shot_experiment "ETTm2" "ETTm1" 7 192 64 True 24
run_zero_shot_experiment "ETTm2" "ETTm1" 7 336 64 True 24
run_zero_shot_experiment "ETTm2" "ETTm1" 7 720 64 True 24
