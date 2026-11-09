#!/bin/bash
# Official SentEval downstream classification reproduction (Table 3 tasks).
#SBATCH --job-name=aoe-senteval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zsx/nlp/results/downstream/slurm-%j.out
#SBATCH --error=/home/zsx/nlp/results/downstream/slurm-%j.err

set -euo pipefail
mkdir -p /home/zsx/nlp/results/downstream
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate nlp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi -L

python /home/zsx/nlp/src/senteval_downstream.py \
  --model /home/zsx/nlp/models/angle-bert-base-uncased-nli-en-v1 \
  --task-path /home/zsx/nlp/data/senteval \
  --output /home/zsx/nlp/results/downstream/official_senteval.json \
  --device cuda \
  --pooling cls_avg \
  --batch-size 64 \
  --max-length 512 \
  --seed 42
