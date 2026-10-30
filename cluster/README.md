# Running GPU stages on Clemson Palmetto

The core pipeline runs on a laptop CPU. These jobs cover the GPU-heavy parts: open-vocabulary
detection with SAM segmentation, a local VLM through Ollama, and Stage 12 label generation.

1. Clone the repository on Palmetto and copy datasets to `/scratch/$USER/...`.
2. `bash cluster/setup_env.sh` once on a login node. Edit `PYTHON_MODULE` if `module avail python`
   shows a different name.
3. `mkdir -p logs && sbatch cluster/evaluate_gpu.slurm /scratch/$USER/my_dataset`
4. `sbatch cluster/vlm_ollama.slurm /scratch/$USER/my_dataset qwen2.5vl:7b`

Add partition or account lines (`#SBATCH --partition=...`, `#SBATCH --account=...`) if your
allocation needs them, and request a specific GPU type with `--gpus=a100:1` when required.
Datasets use the format described in `src/natural_color_charts/evaluation/dataset.py`.
