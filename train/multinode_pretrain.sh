#!/bin/bash
#SBATCH --nodes=2
#SBATCH --account="overtopmedarc"
#SBATCH --partition=h80i
#SBATCH -D .
#SBATCH --ntasks-per-node=1 # number of MP tasks
#SBATCH --gpus-per-task=8 # number of MP tasks
#SBATCH --gpus-per-node=8
#SBATCH --output=pretrain_logs/O-%A-%a.out
#SBATCH --error=pretrain_logs/E-%A-%a.out
#SBATCH --open-mode=append
#SBATCH --exclusive

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 config experiment context_length"
    echo "This script requires exactly 3 arguments."
    exit 1
fi

# ***********
# COMMAND LINE ARGS
# ***********
CONFIG=$1  # Data mixture "reweighting_config"
EXPERIMENT=$2  # Run Name for logging to Wandb
CONTEXT_LENGTH=$3  # 2048
# ***********

# ***********
# MODEL CONFIG
# ***********
SIZE="8"
MODEL="llama3"
if [ $MODEL == "llama3" ]; then
    MODEL_NAME="meta-llama/Meta-Llama-3-8B"
elif [ $MODEL == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen1.5-${SIZE}B"
elif [ $MODEL == "stable" ]; then
    MODEL_NAME="stabilityai/stablelm-${SIZE}b-4e1t"
else
    echo "Unrecognized Model {$MODEL}"
    exit 1
fi
# ***********

# ***********
# HYPER-PARAMS
# ***********

TARGET_BATCH_SIZE=0
PER_DEVICE_BS=0

if [ $CONTEXT_LENGTH -eq 2048 ]; then
    TARGET_BATCH_SIZE=1024
    if [ $MODEL == "llama3" ]; then
        PER_DEVICE_BS=8
        USE_CPU_OFFLOAD=false
    else
        PER_DEVICE_BS=8
        USE_CPU_OFFLOAD=false
    fi
elif [ $CONTEXT_LENGTH -eq 4096 ]; then
    TARGET_BATCH_SIZE=512
    if [ $MODEL == "llama3" ]; then
        PER_DEVICE_BS=8
        USE_CPU_OFFLOAD=false
    elif [ $MODEL == "stable" ]; then
        PER_DEVICE_BS=8
        USE_CPU_OFFLOAD=false
    else
        PER_DEVICE_BS=1
        USE_CPU_OFFLOAD=true
    fi
elif [ $CONTEXT_LENGTH -eq 8192 ]; then
    TARGET_BATCH_SIZE=512
    if [ $MODEL == "llama3" ]; then
        PER_DEVICE_BS=1
        USE_CPU_OFFLOAD=false
    else
        PER_DEVICE_BS=2
        USE_CPU_OFFLOAD=false
        # If size == 7 do the below...
        # PER_DEVICE_BS=1
        # USE_CPU_OFFLOAD=true
    fi
else
    echo "Unrecognized Context Length {$CONTEXT_LENGTH}"
    exit 1
fi

LR=2e-5  # 3e-4
NUM_EPOCHS=4
WARMUP_FRACTION=0.005 # % of training steps over which to warmup lr
GRAD_CLIPPING=true
GRAD_NORM=1.0
# ***********

# ***********
# LOGS & DATA
# ***********
WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="stable-health"
SAVE_STEP_FREQUENCY=5000  # Save and validate every N steps
OUT_DIR="/weka/home-griffin/weights/pretrain/${MODEL}/${EXPERIMENT}"
echo "Will be saving weights to ${OUT_DIR}"
mkdir -p $OUT_DIR
DATASET="/weka/home-griffin/clinical_pile/v1/packed/${CONFIG}_${MODEL}_${CONTEXT_LENGTH}.memmap"
if [ ! -f $DATASET ]; then
  echo "Error: ${DATASET} does not exist."
  exit 1
fi
# Validation dataset
VALIDATION_DATASET="/weka/home-griffin/clinical_instructions/multimedqa/dataset_hf_artificial"
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
if [ "$EFFECTIVE_BATCH_SIZE" -gt "$TARGET_BATCH_SIZE" ]; then
    echo "Exiting because the effective batch size ($EFFECTIVE_BATCH_SIZE) is greater than the target batch size ($TARGET_BATCH_SIZE)."
    exit 1
fi
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
echo CONTEXT_LENGTH=${CONTEXT_LENGTH}
echo TARGET_BATCH_SIZE=${TARGET_BATCH_SIZE}
echo PER_DEVICE_BS=${PER_DEVICE_BS}
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
    --use_gradient_checkpointing true \
    --use_cpu_offload $USE_CPU_OFFLOAD \
    --dataset $DATASET \
    --dataset_for_pretrain_validation $VALIDATION_DATASET \
    --verbose true \
    --lr $LR \
    --save_model true \
    --save_steps $SAVE_STEP_FREQUENCY \
    --save_limit 25 \
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
