#!/bin/bash

set -e

EXPERIMENT=$1
DATASET=$2
IS_COT=$3
# "/weka/home-griffin/weights/pretrain/Qwen/Qwen1.5-0.5B/${EXPERIMENT}/hf" # $BASE_MODEL
BASE_MODEL=$4
PRETRAIN_WEIGHTS=$5

TRAIN_SOURCE_FILTERS=""
VALIDATION_SOURCE_FILTERS=""

# Pick train-validation source filters
if [ $DATASET == "pubmedqa" ]; then
    TRAIN_SOURCE_FILTERS="pubmedqa_labeled,pubmedqa_artificial"
    VALIDATION_SOURCE_FILTERS="pubmedqa_labeled"
    EVAL_INTERVAL=100
    NUM_EPOCHS=1  # Overfits quickly given artificial nature of examples
elif [ $DATASET == "medmcqa" ]; then
    TRAIN_SOURCE_FILTERS="medmcqa"
    VALIDATION_SOURCE_FILTERS="medmcqa"
    EVAL_INTERVAL=500
    NUM_EPOCHS=2
elif [ $DATASET == "medqa" ]; then
    TRAIN_SOURCE_FILTERS="medqa"
    VALIDATION_SOURCE_FILTERS="medqa"
    EVAL_INTERVAL=500
    NUM_EPOCHS=3
else
    echo "Unrecognized Dataset {$DATASET}"
    exit 1
fi

TRAIN_MODE=""
DATA_DIR=""

if [ "${IS_COT}" == "1" ]; then
    TRAIN_MODE="finetune"
    DATA_DIR="/weka/home-griffin/clinical_instructions/multimedqa/dataset_cot_hf_artificial"
else
    TRAIN_MODE="finetune_mcqa"
    DATA_DIR="/weka/home-griffin/clinical_instructions/multimedqa/dataset_hf_artificial"
fi

WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="multimedqa"
# Insert Path to pretrained weights.
# To train from pre-existing model, just point it to HF hub
OUT_DIR="/weka/home-griffin/weights/finetune/${BASE_MODEL}/${EXPERIMENT}"
LR=3e-5
TARGET_BATCH_SIZE=16
PER_DEVICE_BS=1
PER_DEVICE_EVAL_BS=2
NUM_GPUS=8
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
CONTEXT_LENGTH=2048
MAX_VAL_BATCHES=1024

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

echo "Now beginning to train!"
echo "Batch size of ${PER_DEVICE_BS}"
echo "Gradient Accumulation Steps of ${GRAD_ACCUM}"

echo "Will save model weights to ${OUT_DIR}..."
python3 train.py \
--model_name $PRETRAIN_WEIGHTS \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--experiment $EXPERIMENT \
--gradient_accumulation_steps $GRAD_ACCUM \
--batch_size $PER_DEVICE_BS \
--eval_batch_size $PER_DEVICE_EVAL_BS \
--context_length $CONTEXT_LENGTH \
--num_epochs $NUM_EPOCHS \
--train_type full \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--dataset $DATA_DIR \
--train_source_filters $TRAIN_SOURCE_FILTERS \
--validation_source_filters $VALIDATION_SOURCE_FILTERS \
--verbose true \
--lr $LR \
--save_steps $EVAL_INTERVAL \
--save_model true \
--max_val_batches $MAX_VAL_BATCHES \
--lr_scheduler cosine \
--train_mode $TRAIN_MODE \
--log_to wandb \
