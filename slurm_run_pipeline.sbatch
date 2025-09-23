#!/bin/bash -l
#SBATCH -J mdk_pipeline
#SBATCH -A pi_mg269                        # Account
#SBATCH --partition=gpu_h200               # Bouchet H200 partition
#SBATCH --gres=gpu:h200:1   
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

set -eo pipefail
# Avoid set -u here because conda's activate scripts reference unset toolchain
# variables (ADDR2LINE, etc.) during startup, which causes the job to exit early.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
source /home/eqk3/.hpc_env.sh
use_env /home/eqk3/project_pi_mg269/eqk3/.conda-envs/bindcraft-af3 || {
  echo "Failed to activate bindcraft-af3 environment" >&2
  exit 1
}
export PATH="/home/eqk3/project_pi_mg269/eqk3/.conda-envs/bindcraft-af3/lib/python3.9/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

cd /home/eqk3/MDX

python 03_define_epitope.py
python 04_run_bindcraft.py
python 05_convert_to_nanobody.py
python 06_analyze_results.py

./collect_metrics.py
