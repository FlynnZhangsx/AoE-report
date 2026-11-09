#!/bin/bash
# Strict AoE evaluation using the released BERT-base checkpoint.
# Submit from /home/zsx/nlp with: sbatch slurm/strict_aoe_eval.sbatch
#SBATCH --job-name=aoe-strict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zsx/nlp/results/strict/slurm-%j.out
#SBATCH --error=/home/zsx/nlp/results/strict/slurm-%j.err

set -euo pipefail
mkdir -p /home/zsx/nlp/results/strict
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate nlp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "JOB_ID=${SLURM_JOB_ID:-unknown} HOST=$(hostname)"
nvidia-smi -L

python /home/zsx/nlp/src/strict_eval.py \
  --device cuda \
  --model /home/zsx/nlp/models/angle-bert-base-uncased-nli-en-v1 \
  --data-root /home/zsx/nlp/data \
  --output-dir /home/zsx/nlp/results/strict \
  --batch-size 64 \
  --max-length 512
