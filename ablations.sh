#!/bin/bash

EXPERIMENT=$1
WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="stable-health"
SIZE="1.8B"
MODEL="Qwen/Qwen1.5-${SIZE}"
OUT_DIR="/weka/home-griffin/weights/${MODEL}/${EXPERIMENT}"
DATASET="medarc/clinical_pile_v1_minhash_deduped_tokenized"
LR=3e-5
TARGET_BATCH_SIZE=512
PER_DEVICE_BS=1
NUM_GPUS=8
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
CONTEXT_LENGTH=8192

echo "Batch size of ${PER_DEVICE_BS}"
echo "Gradient Accumulation Steps of ${GRAD_ACCUM}"

echo "Will save model weights to ${OUT_DIR}..."
python train.py \
--model_name $MODEL \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--gradient_accumulation_steps $GRAD_ACCUM \
--batch_size $PER_DEVICE_BS \
--context_length $CONTEXT_LENGTH \
--num_epochs 1 \
--train_type full \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--dataset $DATASET \
--profile_memory true \
--verbose true \
--save_model true \
--save_steps 1000 \
--log_to wandb \
