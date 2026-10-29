#!/bin/bash
# Benchmark with OWLv2 detection and SAM segmentation on one GPU.
# Usage: sbatch cluster/evaluate_gpu.slurm /path/to/dataset_dir
#SBATCH --job-name=ncc-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

DATASET=${1:?dataset directory required}
VENV_DIR=${VENV_DIR:-$HOME/venvs/ncc}
# Keep model weights off the small home quota.
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi

ncc evaluate "$DATASET" \
  --detector transformers \
  --set detection.segmenter=sam \
  --set detection.device=cuda \
  --ablation full --ablation no_reliability --ablation generic_sensitivity \
  --output-dir "outputs/evaluation_$(basename "$DATASET")_$SLURM_JOB_ID" \
  --verbose
