#!/bin/bash

set -e

EXPERIMENT=$1

WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="multimedqa"
SIZE="0.5B"
MODEL="Qwen/Qwen1.5-${SIZE}"
OUT_DIR="/weka/home-griffin/weights/finetune/${MODEL}/${EXPERIMENT}"
DATASET="/weka/home-griffin/clinical_instructions/multimedqa"
LR=3e-5
TARGET_BATCH_SIZE=8
PER_DEVICE_BS=1
NUM_GPUS=8
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
CONTEXT_LENGTH=2048

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

echo "Now beginning to train!"
echo "Batch size of ${PER_DEVICE_BS}"
echo "Gradient Accumulation Steps of ${GRAD_ACCUM}"

echo "Will save model weights to ${OUT_DIR}..."
python3 train.py \
--model_name $MODEL \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--experiment $EXPERIMENT \
--gradient_accumulation_steps $GRAD_ACCUM \
--batch_size $PER_DEVICE_BS \
--context_length $CONTEXT_LENGTH \
--num_epochs 5 \
--train_type full \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--dataset $DATASET \
--verbose true \
--lr $LR \
--save_model true \
--save_steps 100 \
--log_to wandb \
--lr_scheduler cosine \
--mode finetune \
