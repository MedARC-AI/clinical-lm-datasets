#!/bin/bash

set -e

MODEL_CLASS=$1
FEWSHOT_N=3

if [ $MODEL_CLASS == "llama2" ]; then
    MODEL_NAME="meta-llama/Llama-2-7b-hf"
elif [ $MODEL_CLASS == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen1.5-7B"
else
    echo "Unrecognized Model {$MODEL_CLASS}"
    exit 1
fi

BASE_CMD="accelerate launch eval_cot.py -eval_pretrained --model_name ${MODEL_NAME} --fewshot_n ${FEWSHOT_N}"

$BASE_CMD --source pubmedqa_labeled
$BASE_CMD --source medqa
$BASE_CMD --source medmcqa
$BASE_CMD --source mmlu
