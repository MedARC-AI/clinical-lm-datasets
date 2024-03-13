#!/bin/bash

set -e

EXPERIMENTS=(
    "all_v1"
    "pubmed_reference_v2"
    "pubmed_clinical_v2"
    "pubmed_heavy_code_v2"
    "pubmed_general_v2"
    "light_pubmed_heavy_clinical_v2"
    "pubmed_light_general_v1"
)

for exp in "${EXPERIMENTS[@]}"
do
    echo "$exp"
    WEIGHT_DIR="/weka/home-griffin/weights/pretrain/Qwen/Qwen1.5-0.5B/${exp}/hf_final"
    if [ ! -d "$WEIGHT_DIR" ]; then
        python3 ckpt_to_hf.py --experiment $exp
    elif [ -d "$WEIGHT_DIR" ]; then
        echo "Directory ${WEIGHT_DIR} already exists."
    fi

    sbatch finetune_slurm.sh $exp "$WEIGHT_DIR"
done