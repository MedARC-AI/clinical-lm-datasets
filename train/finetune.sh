#!/bin/bash

set -e

EXPERIMENT=$1

WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="multimedqa"
SIZE="0.5B"
BASE_MODEL="Qwen/Qwen1.5-${SIZE}"
WEIGHTS=/weka/home-griffin/weights/pretrain/Qwen/Qwen1.5-0.5B/all_v1/hf_3447 # $BASE_MODEL
OUT_DIR="/weka/home-griffin/weights/finetune/${BASE_MODEL}/${EXPERIMENT}"
DATASET="/weka/home-griffin/clinical_instructions/multimedqa/dataset_hf"
LR=3e-5
TARGET_BATCH_SIZE=32
PER_DEVICE_BS=4
NUM_GPUS=8
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
CONTEXT_LENGTH=2048
EVAL_INTERVAL=500
MAX_VAL_BATCHES=1024
NUM_EPOCHS=5

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

echo "Now beginning to train!"
echo "Batch size of ${PER_DEVICE_BS}"
echo "Gradient Accumulation Steps of ${GRAD_ACCUM}"

echo "Will save model weights to ${OUT_DIR}..."
python3 train.py \
--model_name $WEIGHTS \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--experiment $EXPERIMENT \
--gradient_accumulation_steps $GRAD_ACCUM \
--batch_size $PER_DEVICE_BS \
--eval_batch_size 8 \
--context_length $CONTEXT_LENGTH \
--num_epochs $NUM_EPOCHS \
--train_type full \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--dataset $DATASET \
--verbose true \
--lr $LR \
--save_steps $EVAL_INTERVAL \
--save_model true \
--max_val_batches $MAX_VAL_BATCHES \
--lr_scheduler cosine \
--train_mode finetune \
--log_to wandb \
