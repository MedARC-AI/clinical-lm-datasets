#!/bin/bash
#SBATCH --nodes=16
#SBATCH --account="medarc"
#SBATCH --job-name="pubmed_ft"
#SBATCH -D .
#SBATCH --ntasks-per-node=1 # number of MP tasks
#SBATCH --gpus-per-task=8 # number of MP tasks
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/O-%A-%a.out
#SBATCH --error=logs/E-%A-%a.out
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --exclusive

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 config experiment"
    echo "This script requires exactly 2 arguments."
    exit 1
fi

# ***********
# COMMAND LINE ARGS
# ***********
CONFIG=$1  # Data mixture "reweighting_config"
EXPERIMENT=$2 # Run Name for logging to Wandb
# ***********

# ***********
# MODEL CONFIG
# ***********
SIZE="7"
MODEL="llama2"
CONTEXT_LENGTH=4096
if [ $MODEL == "llama2" ]; then
    MODEL_NAME="meta-llama/Llama-2-${SIZE}b-hf"
elif [ $MODEL == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen1.5-${SIZE}B"
else
    echo "Unrecognized Model {$MODEL}"
    exit 1
fi
# ***********

# ***********
# HYPER-PARAMS
# ***********
LR=3e-4
TARGET_BATCH_SIZE=1024
PER_DEVICE_BS=1
NUM_EPOCHS=1
WARMUP_FRACTION=0.005 # % of training steps over which to warmup lr
GRAD_CLIPPING=true
GRAD_NORM=1.0
# ***********

# ***********
# LOGS & DATA
# ***********
WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="stable-health"
SAVE_STEP_FREQUENCY=500  # Save checkpoint every N steps
OUT_DIR="/weka/home-griffin/weights/pretrain/${MODEL}/${EXPERIMENT}"
echo "Will be saving weights to ${OUT_DIR}"
mkdir -p $OUT_DIR
DATASET="/weka/home-griffin/clinical_pile/v1/packed/${CONFIG}_${MODEL}_${CONTEXT_LENGTH}.memmap"
if [ ! -f $DATASET ]; then
  echo "Error: ${DATASET} does not exist."
  exit 1
fi
export HF_HOME="/weka/home-griffin/cache/huggingface"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"
# ***********

# ***********
# MULTINODE SETTINGS
# ***********
export NCCL_DEBUG=WARN #INFO
export NCCL_PROTO=simple
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((${SLURM_NNODES} * ${SLURM_GPUS_PER_NODE}))
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
# ***********

# ***********
# Compute Grad Accum Steps
# ***********
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $WORLD_SIZE))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
echo "GRAD ACCUM=${GRAD_ACCUM}"
# ***********

module load openmpi cuda/12.1

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

export LAUNCHER="torchrun \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_conf=timeout=90 \
    "

export SCRIPT="train.py"
export SCRIPT_ARGS=" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --model_name $MODEL_NAME \
    --output_dir $OUT_DIR \
    --project_name $WANDB_PROJECT \
    --entity $WANDB_ENTITY \
    --experiment $EXPERIMENT \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --batch_size $PER_DEVICE_BS \
    --context_length $CONTEXT_LENGTH \
    --num_epochs $NUM_EPOCHS \
    --train_type full \
    --use_gradient_checkpointing false \
    --use_cpu_offload false \
    --dataset $DATASET \
    --verbose true \
    --lr $LR \
    --save_model true \
    --save_steps $SAVE_STEP_FREQUENCY \
    --log_to wandb \
    --lr_scheduler cosine \
    --train_mode pretrain \
    --world_size=$WORLD_SIZE \
    --warmup_fraction $WARMUP_FRACTION \
    --apply_gradient_clipping $GRAD_CLIPPING \
    --grad_norm $GRAD_NORM \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun --label $CMD
