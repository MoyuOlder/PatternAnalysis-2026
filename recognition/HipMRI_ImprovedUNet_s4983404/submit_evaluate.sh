#!/bin/bash
#SBATCH --job-name=hipmri_eval
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=evaluate_%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
source "$HOME/venv/bin/activate"
for specification in basic:679897 improved:679896; do
  model="${specification%:*}"
  job="${specification#*:}"
  checkpoint="runs/${model}_full_${job}/best.pt"
  python -u train.py --model "$model" --evaluate "$checkpoint" \
    --run "runs/${model}_test_frozen" --split test --workers 0
  python -u analyse_failures.py --checkpoint "$checkpoint" --output "runs/${model}_validation_failures"
done
