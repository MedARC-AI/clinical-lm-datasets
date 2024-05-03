scp -oStrictHostKeyChecking=no pg11b-4-1-hpc4:target_replay_instruct_llama3_8192.memmap .

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
conda init bash
conda create -n clinical-lm-training python=3.10 --yes
conda activate clinical-lm-training
pip install llama-recipes fastcore "transformers!=4.38.*,!=4.39.*" --extra-index-url https://download.pytorch.org/whl/test/cu121
pip install bitsandbytes>=0.43.0
pip install wandb

git clone https://tmabraham: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@github.com/MedARC-AI/clinical-lm-datasets.git

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export OMPI_MCA_mtl_base_erbose=1
export FI_PROVIDER=efa
export NCCL_TREE_THRESHOLD=0

export NCCL_DEBUG=ERROR
export NCCL_SOCKET_TIMEOUT=600000 # Set the timeout to 10 minutes (60000 milliseconds)
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO

export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

mkdir -p weights/llama-3-8b/baseline
