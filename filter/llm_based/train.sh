#!/bin/bash

set -e

EXPERIMENT=$1
DATA_DIR=/weka/home-griffin/quality_filter/dataset_hf
export WANDB_PROJECT=quality-filter
export WANDB_ENTITY=griffin-adams
export WANDB_RUN_ID=$EXPERIMENT
MODEL=FacebookAI/roberta-large # allenai/biomed_roberta_base
OUTPUT_DIR=/weka/home-griffin/weights/quality-filter/$EXPERIMENT
ACCELERATE_CONFIG=/weka/home-griffin/cache/huggingface/accelerate/distill_config.yaml


accelerate launch --config_file $ACCELERATE_CONFIG train_classifier.py \
    --dataset_name $DATA_DIR \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --report_to wandb \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --model_name $MODEL \
    --learning_rate 2e-5 \
    --warmup_steps 200 \
    --evaluation_strategy steps \
    --bf16 \
    --save_total_limit 3 \
    --max_steps 25000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --label_smoothing_factor 0.1 \