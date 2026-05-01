#!/bin/bash
#SBATCH --job-name=VFO_generated_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=13000M
#SBATCH --output=slurm_logs/VFO_generated_gpu_%A_%a.out
#SBATCH --error=slurm_logs/VFO_generated_gpu_%A_%a.err

set -euo pipefail


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

source ~/.bashrc
conda activate desc-env

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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Host: $(hostname)"
echo "Run dir: ${RUN_DIR}"
echo "Start time: $(date)"
echo "============================================================"


#==========
# GPU CHECK
#==========

python - <<'PY'
import jax
print("JAX backend =", jax.default_backend())
print("JAX devices =", jax.devices())
PY


#==========
# RUN
#==========

python driver.py


#==========
# COMPLETE
#==========

echo "============================================================"
echo "Finished task ${SLURM_ARRAY_TASK_ID}"
echo "End time: $(date)"
echo "============================================================"