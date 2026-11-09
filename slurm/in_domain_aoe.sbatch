#!/bin/bash
# In-domain BERT-base AoE experiments for STS-B and GIS.
#SBATCH --job-name=aoe-indomain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zsx/nlp/results/in_domain_strict/slurm-%j.out
#SBATCH --error=/home/zsx/nlp/results/in_domain_strict/slurm-%j.err

set -euo pipefail
mkdir -p /home/zsx/nlp/results/in_domain_strict
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate nlp
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi -L

python /home/zsx/nlp/src/in_domain_train.py \
  --device cuda \
  --model /home/zsx/nlp/models/angle-bert-base-uncased-nli-en-v1 \
  --data-root /home/zsx/nlp/data \
  --output-dir /home/zsx/nlp/results/in_domain_strict \
  --tasks stsb gis \
  --objectives aoe \
  --epochs 5 \
  --learning-rate 5e-5 \
  --seed 42 \
  --stsb-batch-size 32 \
  --gis-batch-size 8 \
  --eval-batch-size 32 \
  --pooling cls_avg \
  --stsb-max-length 128 \
  --gis-max-length 512 \
  --cosine-weight 1.0 \
  --ibn-weight 1.0 \
  --angle-weight 0.02 \
  --cosine-tau 20 \
  --ibn-tau 20 \
  --angle-tau 20 \
  --fp16
