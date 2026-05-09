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
num_workers=${NUM_WORKERS:-8}
learning_rate=${LEARNING_RATE:-0.001}
seq_len=${SEQ_LEN:-512}
train_epochs=${TRAIN_EPOCHS:-10}
seed=${SEED:-2021}
log_to_console=${LOG_TO_CONSOLE:-True}

wandb_project=${WANDB_PROJECT:-time-vlm-ablation-reproduction}
wandb_entity=${WANDB_ENTITY:-}
wandb_mode=${WANDB_MODE:-online}
wandb_group=${WANDB_GROUP:-weather-core-ablation}
wandb_tags=${WANDB_TAGS:-phase3-ablation,weather}
summary_output_dir=${SUMMARY_OUTPUT_DIR:-reports/weather_core_ablation}

percents=${PERCENTS:-"0.1"}
pred_lens=${PRED_LENS:-"96 192 336 720"}
variants=${VARIANTS:-"full no_ral no_ral_l no_ral_g no_val no_tal"}

mkdir -p logs/ablation
echo "Working directory: $(pwd)"
echo "Logs directory: $(pwd)/logs/ablation"

if [ ! -f "./dataset/Weather.csv" ]; then
    echo "Missing ./dataset/Weather.csv. Download the preprocessed datasets and place Weather.csv under ./dataset before running."
    exit 1
fi

weather_d_model() {
    local percent=$1
    local pred_len=$2

    if [ "$percent" = "1" ] || [ "$percent" = "1.0" ]; then
        case "$pred_len" in
            96) echo 64 ;;
            192) echo 64 ;;
            336) echo 128 ;;
            720) echo 64 ;;
            *) echo 64 ;;
        esac
    else
        case "$pred_len" in
            96) echo 128 ;;
            192) echo 128 ;;
            336) echo 256 ;;
            720) echo 256 ;;
            *) echo 128 ;;
        esac
    fi
}

task_for_percent() {
    local percent=$1
    if [ "$percent" = "1" ] || [ "$percent" = "1.0" ]; then
        echo "long_term_forecast"
    else
        echo "few_shot_forecast"
    fi
}

run_weather_ablation() {
    local percent=$1
    local pred_len=$2
    local variant=$3
    local d_model=$4
    local task_name
    task_name=$(task_for_percent "$percent")

    local run_name="weather_${variant}_${percent}p_sl${seq_len}_pl${pred_len}_seed${seed}_${vlm_type}"
    local log_file="logs/ablation/${run_name}.log"

    echo "Running ${run_name}"
    : > "$log_file"

    cmd=(
    python -u run.py
      --task_name "$task_name" \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path Weather.csv \
      --model_id "Weather_${seq_len}_${pred_len}_${percent}p_${variant}" \
      --model "$model_name" \
      --data custom \
      --features M \
      --seq_len "$seq_len" \
      --label_len 48 \
      --pred_len "$pred_len" \
      --d_model "$d_model" \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des "WeatherCoreAblation" \
      --itr 1 \
      --gpu "$gpu" \
      --use_amp \
      --train_epochs "$train_epochs" \
      --image_size "$image_size" \
      --norm_const "$norm_const" \
      --periodicity 144 \
      --three_channel_image "$three_channel_image" \
      --finetune_vlm "$finetune_vlm" \
      --batch_size "$batch_size" \
      --learning_rate "$learning_rate" \
      --num_workers "$num_workers" \
      --vlm_type "$vlm_type" \
      --use_mem_gate True \
      --dropout 0.1 \
      --percent "$percent" \
      --seed "$seed" \
      --ablation_variant "$variant" \
      --use_wandb True \
      --wandb_project "$wandb_project" \
      --wandb_entity "$wandb_entity" \
      --wandb_group "$wandb_group" \
      --wandb_run_name "$run_name" \
      --wandb_tags "$wandb_tags,${task_name},${variant},percent-${percent},pred-len-${pred_len}" \
      --wandb_mode "$wandb_mode"
    )

    echo "Logging to $(pwd)/${log_file}"
    if [ "$log_to_console" = "True" ] || [ "$log_to_console" = "true" ] || [ "$log_to_console" = "1" ]; then
        "${cmd[@]}" 2>&1 | tee "$log_file"
    else
        "${cmd[@]}" > "$log_file" 2>&1
    fi
}

for percent in $percents; do
    for pred_len in $pred_lens; do
        d_model=$(weather_d_model "$percent" "$pred_len")
        for variant in $variants; do
            run_weather_ablation "$percent" "$pred_len" "$variant" "$d_model"
        done
    done
done

python scripts/summarize_forecasting_results.py \
  --results_dir results \
  --output_dir "$summary_output_dir" \
  --method_name "$model_name" \
  --tasks long_term_forecast few_shot_forecast \
  --datasets Weather
