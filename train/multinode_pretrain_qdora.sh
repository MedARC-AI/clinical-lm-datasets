#!/bin/bash
#SBATCH --account=overtopmedarc
#SBATCH -D .
#SBATCH --partition=h80i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gpus-per-node=8
#SBATCH --mem=500gb
#SBATCH --cpus-per-gpu=16
#SBATCH --output=pretrain_logs/O-%A-%a.out
#SBATCH --error=pretrain_logs/E-%A-%a.out
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
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
# ***********

# ***********
# HYPER-PARAMS
# ***********
TARGET_BATCH_SIZE=0
PER_DEVICE_BS=0

if [ $CONTEXT_LENGTH -eq 2048 ]; then
    TARGET_BATCH_SIZE=1024
    PER_DEVICE_BS=8
    USE_CPU_OFFLOAD=false
elif [ $CONTEXT_LENGTH -eq 4096 ]; then
    TARGET_BATCH_SIZE=512
    PER_DEVICE_BS=8
    USE_CPU_OFFLOAD=false
elif [ $CONTEXT_LENGTH -eq 8192 ]; then
    TARGET_BATCH_SIZE=512
    PER_DEVICE_BS=1
    USE_CPU_OFFLOAD=false
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

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# export MASTER_PORT=12340
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

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

echo "Starting python script"

# run setup script to init environment
module load openmpi cuda/12.1

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

# NCCL
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export OMPI_MCA_mtl_base_verbose=1
export FI_PROVIDER=efa
export NCCL_TREE_THRESHOLD=0

export NCCL_DEBUG=ERROR
export NCCL_SOCKET_TIMEOUT=600000 # Set the timeout to 10 minutes (60000 milliseconds)
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO

# export NCCL_IBEXT_DISABLE=1
# export NCCL_SOCKET_IFNAME=^docker0,lo

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
echo "Using python from $(which python)"
echo "Using torch from $(python -c 'import torch; print(torch.__file__)')"
echo "Using torch cuda from $(python -c 'import torch; print(torch.version.cuda)')"
echo "Using nccl from $(python -c 'import torch; print(torch.cuda.nccl.version())')"

# print cuda home
echo "CUDA_HOME=$CUDA_HOME"

srun --label python train_qdora.py \
--world_size $WORLD_SIZE \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
--model_name $MODEL_NAME \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--experiment $EXPERIMENT \
--dataset $DATASET \
--dataset_for_pretrain_validation $VALIDATION_DATASET \
--batch_size $MAX_BATCH_SIZE \
--num_epochs $NUM_EPOCHS \
--context_length $CONTEXT_LENGTH \
--gradient_accumulation_steps $GRAD_ACCUM \
--train_type bnb_dora \
--use_gradient_checkpointing true \
--reentrant_checkpointing true \
--use_activation_cpu_offload false \
--use_cpu_offload $USE_CPU_OFFLOAD \
--log_to wandb \
--verbose true \
--lr $LR \
--save_model true \
--save_steps $SAVE_STEP_FREQUENCY \
--save_limit 25 \
--warmup_fraction $WARMUP_FRACTION \
--apply_gradient_clipping $GRAD_CLIPPING \
--grad_norm $GRAD_NORM \
--train_mode pretrain \