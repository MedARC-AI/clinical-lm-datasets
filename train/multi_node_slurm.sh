#!/bin/bash
#SBATCH --job-name="pubmed_ft"
#SBATCH --nodes=2
#SBATCH --output=logs/latest_%A_%a.out  # Set this dir where you want slurm outs to go
#SBATCH --partition=a40
#SBATCH --account="topmedarc"
#SBATCH --open-mode=append
#SBATCH --exclusive

CONFIG=$1
EXPERIMENT=$2

module load openmpi cuda/12.1

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

export NCCL_DEBUG=INFO
# export NCCL_PROTO=simple

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
# export NCCL_TREE_THRESHOLD=0

export WANDB_MODE="online"
export WANDB_START_METHOD="thread"

export TORCH_SHOW_CPP_STACKTRACES=1

# sent to sub script
export MASTER_PORT=12802
export I_MPI_PORT_RANGE=12800:12804
export I_MPI_HYDRA_BOOTSTRAP=ssh

echo go $COUNT_NODE
echo $HOSTNAMES
echo $MASTER_ADDR
echo $I_MPI_PORT_RANGE

export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'
export HF_HOME="/weka/home-griffin/cache/huggingface"
export TRANSFORMERS_CACHE="/weka/home-griffin/cache/huggingface/models"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

# wandb login --cloud --relogin $WANDB_API_KEY
srun --label multi_node.sh $CONFIG $EXPERIMENT
