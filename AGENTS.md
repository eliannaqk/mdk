# Repository Guidelines

##need to know
- monitor job status with squeue -u eqk3
- when reporting outputs alway reference which scripts were used to create the data (and path) and save outputs in csv in `/metrics`
- after sbatch submittion give me a summary of what you did, what scripts are being run with what data, and where we can find the outputs
- don't push to github unless instructed
- always write agent summary to summary.text 
- at the begining of each initialization read most recent summary (end portion of file) 

##Environment Policy
- always use bindcraft-af3 or oc-opencrispr or bio-utils (bioinformatics tools) or oc-opencrispr-esm (for gpu with esmfold) or oc-af3 (for gpu with alphafold3 and JAX)
- If neither of these environments are available, the agent should fail the task rather than attempting to create a new, unapproved environment.

### Agent startup: ensure Conda env is active
- Non-interactive shells (e.g., SLURM, CI) do not auto-load Conda. Always initialize Conda, then activate the approved env before running any tool:
- Interactive login shells can alternatively auto-activate via ~/.bashrc:
```bash
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
source /home/eqk3/.hpc_env.sh
use_env /home/eqk3/project_pi_mg269/eqk3/.conda-envs/   # or oc-af3-clone, etc.
  ```
### SLURM script requirements (agent must include these)
```bash
#!/bin/bash -l
#SBATCH -J <jobname>
#SBATCH -p <partition>
#SBATCH --gres=gpu:<type>:<n>
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
source /home/eqk3/.hpc_env.sh
conda activate bindcraft-af3 # or oc-af3, oc-opencrispr-esm, or another
 #move to project directory
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

```
- The agent must prefer `#!/bin/bash -l`
- to create new env, always place in proj
```bash
PROJ=/home/eqk3/project_pi_mg269/eqk3
source ~/miniconda3/etc/profile.d/conda.sh. 
conda create -y -p $PROJ/.conda-envs/
```

## Cluster & GPU Usage
- Interactive GPU shells:
  - H200: `srun -p gpu_h200 --gres=gpu:h200:1 -t 4:00:00 --pty bash -l`
  - Dev queue: `srun -p gpu_devel --gres=gpu:1 -t 4:00:00 --pty bash -l`
- Inside the session, run `conda activate oc-opencrispr`, or `oc-af3` then execute training/eval commands.
- Data location: `/home/eqk3/project_pi_mg269/eqk3/`

## Security & Configuration Tips
- CUDA/Torch: wheels resolve via `pyproject.toml` (CUDA 12.4 extra index). Use compatible drivers.

## Weights & Biases Setup
- Login interactively: `conda activate oc-opencrispr` then `wandb login`.
- Non-interactive jobs: prefer prior login so tokens are cached; if necessary, export `WANDB_API_KEY`.
- Defaults: trainer logs to project `MDK`, entity `eqk3` 

## Project Structure & Module Organization
The root hosts stepwise pipeline scripts (`03_define_epitope.py`, `04_run_bindcraft.py`, `05_convert_to_nanobody.py`, `06_analyze_results.py`) plus orchestration helpers such as `batch_design.py` and `export_results.py`. Core reusable logic lives in `src/`, with preparation, design, formatting, and validation modules grouped under `src/utils/` for scoring and structural helpers. Experimental data, templates, and downloaded PDB files land in `data/`, while generated sequences, models, and reports should stay under `results/` in the matching subfolder created by each run. Tests reside in `test/`, and configuration defaults are captured in `config.yml` (create a `config.yaml` copy when running legacy scripts that demand that extension).

## Build, Test, and Development Commands
Use Python 3.9 within the `bindcraft-af3` conda environment described in `requirements_and_setup_files.sh`. Populate structural inputs via `python download_structures.py`. Run the stepwise pipeline with the numbered scripts or launch parallel campaigns with `python batch_design.py --config config.yml`. Execute quick functional checks using `python src/design_binders.py --help` to confirm CLI wiring. 

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and type hints on all public functions. Prefer descriptive module-level docstrings and keep functions short, composable, and side-effect aware. Name files and variables with snake_case; reserve PascalCase for classes. When adding configs, mirror the existing nested key style in `config.yml` and keep paths relative to the repository root. Use `black` and `ruff` if available; otherwise run `python -m compileall` for syntax validation.

## Testing Guidelines
Write pytest-compatible tests under `test/` using filenames like `test_<feature>.py`. Validate structural utilities and scoring routines with lightweight fixtures to avoid large downloads. Run `pytest test -v` before submitting changes; integrate new datasets by mocking filesystem interactions where possible. Target deterministic outputs so CI can rely on CPU-only runs even when GPU toolchains are absent.
