
#!/bin/bash

echo $SLURM_NODEID
echo $SLURM_PROCID

# CONFIG=$1
# EXPERIMENT=$2

# WANDB_ENTITY="griffin-adams"
# WANDB_PROJECT="stable-health"
# SIZE="0.5B"
# MODEL="Qwen/Qwen1.5-${SIZE}"
# OUT_DIR="/weka/home-griffin/weights/pretrain/${MODEL}/${EXPERIMENT}"
# DATASET="/weka/home-griffin/clinical_pile/v1/tokenized/dataset_hf_${CONFIG}"
# LR=3e-5
# TARGET_BATCH_SIZE=512
# PER_DEVICE_BS=2
# NUM_GPUS=8
# EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS))
# GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
# CONTEXT_LENGTH=8192

# head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# master_port=29501

# module load openmpi cuda/12.1

# cd /weka/home-griffin/clinical-lm-datasets/train
# source /weka/home-griffin/envs/train/bin/activate

# export GPUS_PER_NODE=8
# export HF_HOME="/weka/home-griffin/cache/huggingface"
# export TRANSFORMERS_CACHE="/weka/home-griffin/cache/huggingface/models"
# export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

# echo "Slurm Process ID --> ${SLURM_PROCID}"
# echo "Slurm Node ID --> ${SLURM_NODEID}"

# export LAUNCHER="accelerate launch \
#     --config_file /weka/home-griffin/cache/huggingface/accelerate/two_node.yaml
#     --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
#     --num_machines $SLURM_NNODES \
#     --rdzv_backend c10d \
#     --main_process_ip $head_node_ip \
#     --main_process_port $master_port \
#     --machine_rank $SLURM_NODEID \
#     "
# export SCRIPT="train.py"
# export SCRIPT_ARGS=" \
#     --master_addr="localhost" \
#     --master_port="5321${SLURM_NODEID}" \
#     --model_name $MODEL \
#     --output_dir $OUT_DIR \
#     --project_name $WANDB_PROJECT \
#     --entity $WANDB_ENTITY \
#     --experiment $EXPERIMENT \
#     --gradient_accumulation_steps $GRAD_ACCUM \
#     --batch_size $PER_DEVICE_BS \
#     --context_length $CONTEXT_LENGTH \
#     --num_epochs 1 \
#     --train_type full \
#     --use_gradient_checkpointing false \
#     --use_cpu_offload false \
#     --dataset $DATASET \
#     --verbose true \
#     --lr $LR \
#     --save_model true \
#     --save_steps 500 \
#     --log_to wandb \
#     --lr_scheduler cosine \
#     --train_mode pretrain \
#     "

# export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
# $CMD