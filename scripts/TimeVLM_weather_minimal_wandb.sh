#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false

model_name=${MODEL_NAME:-TimeVLM}
vlm_type=${VLM_TYPE:-clip}
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

wandb_project=${WANDB_PROJECT:-time-vlm-minimal-reproduction}
wandb_entity=${WANDB_ENTITY:-}
wandb_mode=${WANDB_MODE:-online}
wandb_group=${WANDB_GROUP:-weather-minimal-reproduction}
wandb_tags=${WANDB_TAGS:-minimal-reproduction,weather}
summary_output_dir=${SUMMARY_OUTPUT_DIR:-reports/weather_minimal_wandb}

percents=${PERCENTS:-"0.1 1"}
pred_lens=${PRED_LENS:-"96 192 336 720"}

mkdir -p logs

weather_d_model() {
    local percent=$1
    local pred_len=$2

    if [ "$percent" = "1" ]; then
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

run_weather() {
    local percent=$1
    local pred_len=$2
    local d_model=$3
    local task_name="few_shot_forecast"

    if [ "$percent" = "1" ]; then
        task_name="long_term_forecast"
    fi

    local run_name="weather_${percent}p_sl${seq_len}_pl${pred_len}_seed${seed}_${vlm_type}"
    local log_file="logs/${run_name}.log"

    echo "Running ${run_name}"

    python -u run.py \
      --task_name "$task_name" \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path Weather.csv \
      --model_id "Weather_${seq_len}_${pred_len}_${percent}p" \
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
      --des "WeatherMinimalWandb" \
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
      --use_wandb True \
      --wandb_project "$wandb_project" \
      --wandb_entity "$wandb_entity" \
      --wandb_group "$wandb_group" \
      --wandb_run_name "$run_name" \
      --wandb_tags "$wandb_tags,${task_name},percent-${percent},pred-len-${pred_len}" \
      --wandb_mode "$wandb_mode" > "$log_file" 2>&1
}

for percent in $percents; do
    for pred_len in $pred_lens; do
        d_model=$(weather_d_model "$percent" "$pred_len")
        run_weather "$percent" "$pred_len" "$d_model"
    done
done

python scripts/summarize_forecasting_results.py \
  --results_dir results \
  --output_dir "$summary_output_dir" \
  --method_name "$model_name" \
  --tasks long_term_forecast few_shot_forecast \
  --datasets Weather
