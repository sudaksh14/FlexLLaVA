#!/bin/bash
# Submit I-3 stages B -> C -> D as one dependency chain (run from the login node; it only calls sbatch).
#   scripts/croppilot/run_selfroi_chain.sh <stage-A job ids, colon separated, e.g. 27920:27921>
# Stage A (selfroi_guide.sh, GUIDE=3 and GUIDE=0) must already be submitted.
set -euo pipefail
A_JOBS=${1:?stage A job ids, colon separated}
cd /home/skalra/FlexLLaVA

B=$(sbatch --parsable --dependency=afterok:${A_JOBS} scripts/croppilot/selfroi_boxes.sh)
echo "stage B: $B"

# Stage C: box source x input mode, at tok_level indices 0 (256), 2 (64), 3 (16).
C_JOBS=""
for spec in "g16 crop" "g16 globalcrop" "g256 crop" "centre crop" "centre globalcrop" "gtctr crop" "random crop"; do
    set -- $spec
    J=$(SRC_TAG=$1 MODE=$2 sbatch --parsable --dependency=afterok:${B} --job-name=cp_$1_$2 --array=0,2,3 eval_croppilot_level.sh)
    echo "stage C $1 $2: $J"
    C_JOBS="${C_JOBS:+$C_JOBS:}$J"
done

D=$(sbatch --parsable --dependency=afterok:${C_JOBS} scripts/croppilot/analyze_selfroi.sh)
echo "stage D: $D"
