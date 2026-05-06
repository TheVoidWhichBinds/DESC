#!/bin/bash

#SBATCH --account=torch_pr_292_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/dz2596/DESC/research/lit_comp/tutorials/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/dz2596/DESC/research/lit_comp/tutorials/slurm_logs/%x_%A_%a.err

#==============================================================================================================
# One array task = one tolerance combination = one GPU.
#
# Submit through:
#   bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh <tutorial>
#==============================================================================================================

set -euo pipefail

if [ -z "${DESC_TUTORIAL:-}" ]; then
    echo "Error: DESC_TUTORIAL is unset."
    echo "Submit with: bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh basic_qs"
    exit 1
fi

if [ -z "${DESC_CASE_START_INDEX:-}" ]; then
    echo "Error: DESC_CASE_START_INDEX is unset."
    echo "Submit with: bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh ${DESC_TUTORIAL}"
    exit 1
fi

cd /scratch/dz2596/DESC

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/dz2596/conda-envs/desc

unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DESC_BACKEND=gpu

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p research/lit_comp/tutorials/slurm_logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Tutorial: ${DESC_TUTORIAL}"
echo "Host: $(hostname)"
echo "DESC_CASE_START_INDEX: ${DESC_CASE_START_INDEX}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

python3 - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

python3 research/lit_comp/tutorials/driver.py \
    --tutorial "${DESC_TUTORIAL}" \
    --sweep-index "${SLURM_ARRAY_TASK_ID}" \
    --case-start-index "${DESC_CASE_START_INDEX}"
