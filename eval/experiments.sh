
#!/bin/bash

set -e

MODEL_CLASS=$1
TYPE=$2
SIZE=$3
EXPERIMENT=$4

FT_DIR=/weka/home-griffin/weights/finetune
PT_DIR=/weka/home-griffin/weights/pretrain
 
FEWSHOT_DEFAULT=3

BASE_ARGS="--sources all --fewshot_n $FEWSHOT_DEFAULT"
MODEL_ARGS="--model_path TBD"
OVERWRITE_ARGS=""  # Set to -overwrite overwrite
SHOULD_COT=true
SHOULD_LL=true

if [ $TYPE == "base" ]; then
    if [ $MODEL_CLASS == "qwen" ]; then
        MODEL_ARGS="--model_path Qwen/Qwen1.5-${SIZE}B"
    elif [ $MODEL_CLASS == "llama2" ]; then
        MODEL_ARGS="--model_path meta-llama/Llama-2-${SIZE}b-hf"
    elif [ $MODEL_CLASS == "gemma" ]; then
        MODEL_ARGS="--model_path google/gemma-${SIZE}b"
    else
        echo "Huh?"
    fi
elif [ $TYPE == "pt" ]; then
    MODEL_ARGS="--model_path ${PT_DIR}/${MODEL_CLASS}/${EXPERIMENT}"
else  # Been fine-tuned
    MODEL_ARGS="--model_path ${FT_DIR}/${MODEL_CLASS}/${EXPERIMENT}"
    # Sources are going to depend on what it was trained on
    BASE_ARGS="--sources xxx --fewshot_n 0"
fi

echo "Starting evals with ${BASE_ARGS} ${MODEL_ARGS}"
if [ $SHOULD_LL ]; then
    python3 eval_vllm.py $BASE_ARGS $MODEL_ARGS $OVERWRITE_ARGS
fi

if [ $SHOULD_COT ]; then
    python3 eval_vllm.py $BASE_ARGS $MODEL_ARGS $OVERWRITE_ARGS -cot
fi
