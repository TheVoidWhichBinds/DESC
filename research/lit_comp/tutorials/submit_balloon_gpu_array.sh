#!/bin/bash

#==============================================================================================================
# Submit the balloon tolerance sweep as one GPU SLURM array.
#==============================================================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/submit_tutorial_gpu_array.sh" "balloon"
