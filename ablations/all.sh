# !/bin/bash
# SBATCH --job-name=ablation-all
# SBATCH --partition=a80
# SBATCH --nodes=1
# SBATCH --gpus=8
# SBATCH --exclusive
# SBATCH --account=overtopmedarc
# SBATCH --mem=300gb
# SBATCH --requeue

source /etc/profile.d/modules.sh
module load cuda/12.1

source /weka/home-griffin/data/bin/activate

export HF_HOME="/weka/home-griffin/cache/huggingface"
export TRANSFORMERS_CACHE="/weka/home-griffin/cache/huggingface/models"
export HF_DATASETS_CACHE="/weka/home-griffin/cache/huggingface/datasets/"

cd /weka/home-griffin/clinical-lm-datasets/ablations
srun run.sh all
