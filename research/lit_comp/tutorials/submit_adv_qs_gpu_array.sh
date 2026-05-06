#!/bin/bash

#==============================================================================================================
# Submit the adv_qs tolerance sweep as one GPU SLURM array.
#==============================================================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/submit_tutorial_gpu_array.sh" "adv_qs"
