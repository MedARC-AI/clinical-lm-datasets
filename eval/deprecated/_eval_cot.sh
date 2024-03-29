#!/bin/bash

set -e

MODEL_CLASS=$1
FEWSHOT_N=$2
DATASET=$3
CKPT=-1


if [ "$FEWSHOT_N" -eq 0 ]; then # Use Finetuned for zeroshot
    if [ $MODEL_CLASS == "llama2" ]; then
        WEIGHT_DIR="/weka/home-griffin/weights/finetune/meta-llama/Llama-2-7b-hf"
        MODEL_NAME="meta-llama/Llama-2-7b-hf"
        EXPERIMENT="llama2_7b_pubmed_medmcqa"
    elif [ $MODEL_CLASS == "qwen" ]; then
        WEIGHT_DIR="/weka/home-griffin/weights/finetune/Qwen/Qwen1.5-7B"
        MODEL_NAME="Qwen/Qwen1.5-7B"
        EXPERIMENT="qwen_7b_base_medmcqa"
    else
        echo "Unrecognized Model {$MODEL_CLASS}"
        exit 1
    fi
else  # Use "Pretrained models for fewshot"
    if [ $MODEL_CLASS == "llama2" ]; then
        WEIGHT_DIR="/weka/home-griffin/weights/pretrain/llama2"
        MODEL_NAME="meta-llama/Llama-2-7b-hf"
        EXPERIMENT="pubmed_llama2_7b_4k"
    elif [ $MODEL_CLASS == "qwen" ]; then
        WEIGHT_DIR="/weka/home-griffin/weights/pretrain/qwen"
        MODEL_NAME="Qwen/Qwen1.5-7B"
        EXPERIMENT="pubmed_qwen_7b_2k"
    else
        echo "Unrecognized Model {$MODEL_CLASS}"
        exit 1
    fi
fi

accelerate launch eval_cot.py \
    --source ${DATASET} \
    --weight_dir ${WEIGHT_DIR} \
    --experiment ${EXPERIMENT} \
    --model_name ${MODEL_NAME} \
    --fewshot_n ${FEWSHOT_N} \
    --ckpt ${CKPT} \
