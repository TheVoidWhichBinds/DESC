#!/bin/bash

#==============================================================================================================
# Submit one DESC tutorial tolerance sweep as a GPU SLURM array.
#
# Usage from DESC root:
#   bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh basic_qs
#   bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh adv_qs
#   bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh balloon
#   bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh neoclassical
#
# Optional overrides:
#   DESC_GPU_TIME=06:00:00 DESC_GPU_MEM=40G bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh balloon
#==============================================================================================================

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Error: provide exactly one tutorial name."
    echo "Usage: bash research/lit_comp/tutorials/submit_tutorial_gpu_array.sh basic_qs"
    exit 1
fi

TUTORIAL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_DIR="${SCRIPT_DIR}/${TUTORIAL}"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"

if [ ! -d "${TUTORIAL_DIR}" ]; then
    echo "Error: tutorial directory does not exist: ${TUTORIAL_DIR}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

case "${TUTORIAL}" in
    basic_qs)
        DEFAULT_SWEEP_COUNT=36
        DEFAULT_TIME="03:00:00"
        DEFAULT_MEM="30G"
        ;;

    adv_qs)
        DEFAULT_SWEEP_COUNT=36
        DEFAULT_TIME="04:00:00"
        DEFAULT_MEM="30G"
        ;;

    balloon)
        DEFAULT_SWEEP_COUNT=36
        DEFAULT_TIME="04:00:00"
        DEFAULT_MEM="30G"
        ;;

    neoclassical)
        DEFAULT_SWEEP_COUNT=36
        DEFAULT_TIME="04:00:00"
        DEFAULT_MEM="30G"
        ;;

    *)
        echo "Error: unknown tutorial: ${TUTORIAL}"
        echo "Known tutorials: basic_qs, adv_qs, balloon, neoclassical"
        exit 1
        ;;
esac

SWEEP_COUNT="${DESC_GPU_SWEEP_COUNT:-${DEFAULT_SWEEP_COUNT}}"
TIME_LIMIT="${DESC_GPU_TIME:-${DEFAULT_TIME}}"
MEMORY="${DESC_GPU_MEM:-${DEFAULT_MEM}}"

CASE_START_INDEX="$({
    TUTORIAL_DIR="${TUTORIAL_DIR}" python3 - <<'PY'
from pathlib import Path
import os

tutorial_dir = Path(os.environ["TUTORIAL_DIR"])
existing_indices = [
    int(path.name)
    for path in tutorial_dir.iterdir()
    if path.is_dir() and path.name.isdigit()
]

print(max(existing_indices, default = 0) + 1)
PY
})"

echo "Submitting ${TUTORIAL} GPU array."
echo "First output folder index: ${CASE_START_INDEX}"
echo "Array range: 1-${SWEEP_COUNT}"
echo "Time limit: ${TIME_LIMIT}"
echo "Memory: ${MEMORY}"

sbatch \
    --job-name="${TUTORIAL}_gpu" \
    --array="1-${SWEEP_COUNT}" \
    --time="${TIME_LIMIT}" \
    --mem="${MEMORY}" \
    --export=ALL,DESC_TUTORIAL="${TUTORIAL}",DESC_CASE_START_INDEX="${CASE_START_INDEX}" \
    "${SCRIPT_DIR}/tutorial_gpu_array.sbatch"
