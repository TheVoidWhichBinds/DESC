#!/bin/bash
#SBATCH --job-name=generated_surface_loop
#SBATCH --account=torch_pr_292_courant
#SBATCH --partition=l40s_courant
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=15000M
#SBATCH --output=slurm_logs/generated_surface_loop_%A_%a.out
#SBATCH --error=slurm_logs/generated_surface_loop_%A_%a.err

set -eo pipefail


#==========
# PATHS
#==========

REPO_DIR="/scratch/dz2596/DESC"
RUN_DIR="${REPO_DIR}/research/VFOs"

cd "${RUN_DIR}"

mkdir -p slurm_logs


#==========
# ENVIRONMENT
#==========

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/dz2596/conda-envs/desc

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

unset JAX_PLATFORM_NAME

export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false


#==========
# RUN INFO
#==========

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Repo dir: ${REPO_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Python: $(which python)"
echo "Conda env: ${CONDA_PREFIX}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "JAX_PLATFORM_NAME: ${JAX_PLATFORM_NAME:-unset}"
echo "JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo "============================================================"


#==========
# RUN
#==========
python driver.py --task-id "${SLURM_ARRAY_TASK_ID}"
