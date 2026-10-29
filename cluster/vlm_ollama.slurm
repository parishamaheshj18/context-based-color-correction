#!/bin/bash
# Run a local VLM (Ollama) on a GPU node, then benchmark with scene context and export
# Stage 12 reliability labels.
# Usage: sbatch cluster/vlm_ollama.slurm /path/to/dataset_dir [model]
#SBATCH --job-name=ncc-vlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

DATASET=${1:?dataset directory required}
MODEL=${2:-qwen2.5vl:7b}
VENV_DIR=${VENV_DIR:-$HOME/venvs/ncc}
OLLAMA_DIR=${OLLAMA_DIR:-$HOME/ollama}
export OLLAMA_MODELS=${OLLAMA_MODELS:-/scratch/$USER/ollama_models}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export OLLAMA_HOST=127.0.0.1:$((11000 + SLURM_JOB_ID % 1000))

if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  mkdir -p "$OLLAMA_DIR"
  curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C "$OLLAMA_DIR"
fi
"$OLLAMA_DIR/bin/ollama" serve > "logs/ollama-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID' EXIT
until "$OLLAMA_DIR/bin/ollama" list > /dev/null 2>&1; do sleep 2; done
"$OLLAMA_DIR/bin/ollama" pull "$MODEL"

source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"
COMMON=(--vlm ollama --set "context.ollama.model=$MODEL" --set "context.ollama.host=http://$OLLAMA_HOST"
        --detector transformers --set detection.segmenter=sam --set detection.device=cuda)

ncc evaluate "$DATASET" "${COMMON[@]}" --ablation full --ablation no_context \
  --output-dir "outputs/evaluation_vlm_$SLURM_JOB_ID"
ncc label-references "$DATASET" "${COMMON[@]}" --output-dir "outputs/labels_$SLURM_JOB_ID"
