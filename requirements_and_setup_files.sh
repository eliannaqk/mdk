#!/bin/bash
# 01_setup_environment.sh - Setup script for MDK-LRP1 nanobody design pipeline

echo "========================================"
echo "Setting up MDK-LRP1 Nanobody Design Environment"
echo "========================================"

# Create conda environment
echo "Creating conda environment..."
conda create -n mdk-nanobody python=3.9 -y

# Activate environment
echo "Activating environment..."
conda activate mdk-nanobody

# Install PyTorch (for BindCraft)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install BindCraft
echo "Installing BindCraft..."
git clone https://github.com/martinpacesa/BindCraft.git
cd BindCraft
pip install -e .
cd ..

# Install AlphaFold dependencies
echo "Setting up AlphaFold dependencies..."
pip install jax[cuda]==0.3.25
pip install ml-collections
pip install dm-haiku
pip install optax

# Install ProteinMPNN
echo "Installing ProteinMPNN..."
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN
pip install -e .
cd ..

# Download required models
echo "Downloading models..."
mkdir -p models

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/structures
mkdir -p data/epitopes
mkdir -p data/templates
mkdir -p results/designs
mkdir -p results/nanobodies
mkdir -p results/validation
mkdir -p results/visualizations
mkdir -p notebooks

echo "========================================"
echo "Setup complete!"
echo "Activate environment with: conda activate mdk-nanobody"
echo "Run pipeline starting with: python scripts/02_download_structures.py"
echo "========================================"

# ====== requirements.txt content ======
cat > requirements.txt << 'EOF'
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Bioinformatics
biopython>=1.79
pymol-open-source>=2.5.0

# Structure manipulation
mdtraj>=1.9.7
prody>=2.0

# Deep learning
torch>=2.0.0
einops>=0.6.0

# Protein design tools
pyrosetta  # Requires license from https://www.pyrosetta.org/

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
py3Dmol>=2.0.0
nglview>=3.0.0

# Jupyter support
jupyter>=1.0.0
ipywidgets>=8.0.0

# File handling
pyyaml>=6.0
h5py>=3.7.0

# Web requests
requests>=2.28.0

# Progress bars
tqdm>=4.64.0

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0
EOF

# ====== environment.yml content ======
cat > environment.yml << 'EOF'
name: mdk-nanobody
channels:
  - conda-forge
  - bioconda
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - pytorch>=2.0.0
  - pytorch-cuda=11.8
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - pip
  - pip:
    - biopython
    - pymol-open-source
    - mdtraj
    - prody
    - einops
    - py3Dmol
    - nglview
    - pyyaml
    - h5py
    - requests
    - tqdm
    - pytest
    - pytest-cov
    - jax[cuda]
    - ml-collections
    - dm-haiku
    - optax
EOF