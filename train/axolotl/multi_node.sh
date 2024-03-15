#!/bin/bash
#sleep 30
#fi_info -p efa -t FI_EP_RDMH=hostname

echo myuser=whoami
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc which mpicc
echo HOSTNAMES = $SLURM_JOB_NODELIST
echo hostname = hostname
echo MASTER_ADDR= $SLURM_LAUNCH_NODE_IPADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_JOB_NODELIST= $SLURM_JOB_NODELIST
echo SLURM_JOB_NUM_NODES= $SLURM_JOB_NUM_NODES
echo NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING

export H=hostname
export THEID=$SLURM_PROCID
echo THEID=$THEID

CONFIG=$1
EXPERIMENT=$2

WANDB_ENTITY="griffin-adams"
WANDB_PROJECT="stable-health"
SIZE="0.5B"
MODEL="Qwen/Qwen1.5-${SIZE}"
OUT_DIR="/weka/home-griffin/weights/pretrain/${MODEL}/${EXPERIMENT}"
DATASET="/weka/home-griffin/clinical_pile/v1/tokenized/dataset_hf_${CONFIG}"
LR=3e-5
TARGET_BATCH_SIZE=512
PER_DEVICE_BS=2
NUM_GPUS=8
EFFECTIVE_BATCH_SIZE=$(($PER_DEVICE_BS * $NUM_GPUS * $SLURM_JOB_NUM_NODES))
GRAD_ACCUM=$(($TARGET_BATCH_SIZE / $EFFECTIVE_BATCH_SIZE))
CONTEXT_LENGTH=8192

cd /weka/home-griffin/clinical-lm-datasets/train
source /weka/home-griffin/envs/train/bin/activate

echo "Now beginning to train!"
echo "Batch size of ${PER_DEVICE_BS}"
echo "Gradient Accumulation Steps of ${GRAD_ACCUM}"

echo "Will save model weights to ${OUT_DIR}..."

accelerate launch \
--num_processes=$(( 8 * $SLURM_JOB_NUM_NODES )) \
--num_machines $SLURM_JOB_NUM_NODES \
--machine_rank $THEID \
--main_process_ip $SLURM_LAUNCH_NODE_IPADDR \
--main_process_port $MASTER_PORT \
--mixed_precision=bf16  \
--config_file accelerate_config_debug.yaml \
train.py \
--model_name $MODEL \
--output_dir $OUT_DIR \
--project_name $WANDB_PROJECT \
--entity $WANDB_ENTITY \
--experiment $EXPERIMENT \
--gradient_accumulation_steps $GRAD_ACCUM \
--batch_size $PER_DEVICE_BS \
--context_length $CONTEXT_LENGTH \
--num_epochs 1 \
--train_type full \
--use_gradient_checkpointing false \
--use_cpu_offload false \
--dataset $DATASET \
--verbose true \
--lr $LR \
--save_model true \
--save_steps 500 \
--log_to wandb \
--lr_scheduler cosine \
--train_mode pretrain \
