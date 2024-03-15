#!/bin/bash

set -e

CONFIG=$1
EXPERIMENT=$2

SIZE="7"
CONTEXT_LENGTH=2048
MODEL="llama2"

if [ $MODEL == "llama2" ]; then
    MODEL_NAME="meta-llama/Llama-2-${SIZE}b"
elif [ $MODEL == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen1.5-${SIZE}B"
else
    echo "Unrecognized Model {$MODEL}"
    exit 1
fi

LR=3e-4
TARGET_BATCH_SIZE=512
PER_DEVICE_BS=1
NUM_GPUS=8
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
OUT_DIR="/weka/home-griffin/weights/pretrain/${MODEL}/${EXPERIMENT}"
DATASET="/weka/home-griffin/clinical_pile/v1/packed/${CONFIG}_${MODEL}_${CONTEXT_LENGTH}"
WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="stable-health"

# echo "Generating the ablation dataset if it does not exist..."
# source /weka/home-griffin/envs/data/bin/activate
# cd /weka/home-griffin/clinical-lm-datasets/tokenize
# python3 build_dataset.py --reweighting_config $CONFIG

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

echo "Now beginning to train!"
echo "Batch size of ${PER_DEVICE_BS}"
echo "Gradient Accumulation Steps of ${GRAD_ACCUM}"

echo "Will save model weights to ${OUT_DIR}..."
python3 train.py \
--model_name $MODEL_NAME \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--experiment $EXPERIMENT \
--gradient_accumulation_steps $GRAD_ACCUM \
--batch_size $PER_DEVICE_BS \
--context_length $CONTEXT_LENGTH \
--num_epochs 1 \
--train_type full \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--dataset $DATASET \
--verbose true \
--lr $LR \
--save_model true \
--save_steps 500 \
--log_to wandb \
--lr_scheduler cosine \
--train_mode pretrain \
--warmup_fraction 0.005 \  # https://arxiv.org/pdf/2403.08763.pdf - recommend 0.5% of training steps
--apply_gradient_clipping true \
--grad_norm 1.0 \
