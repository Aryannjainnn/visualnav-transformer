#!/bin/bash
# ============================================================
#  AWS Setup Script for NoMaD Fine-tuning
#  Tested on: Ubuntu 20.04 / 22.04 + CUDA 11.8
#  Run from the root of the repo: bash aws_setup.sh
# ============================================================

set -e  # exit on any error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Repo root: $REPO_ROOT"

# ─── 1. Install system dependencies ──────────────────────────────────────────
echo ""
echo "==> [1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev   # needed by opencv

# ─── 2. Install Miniconda (skip if conda already present) ────────────────────
echo ""
echo "==> [2/6] Checking conda..."
if ! command -v conda &>/dev/null; then
    echo "     conda not found — installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "     Miniconda installed. Re-source your shell or run: source ~/.bashrc"
else
    echo "     conda found: $(conda --version)"
fi

# ─── 3. Create / update the training conda environment ───────────────────────
echo ""
echo "==> [3/6] Creating conda environment (nomad_train)..."
conda env create -f "$REPO_ROOT/train/train_environment.yml" || \
    conda env update -f "$REPO_ROOT/train/train_environment.yml" --prune
echo "     Environment ready."

# Activate the env for the rest of this script
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nomad_train

# ─── 4. Upgrade torch for modern CUDA (AWS typically has CUDA 11.8 / 12.x) ───
echo ""
echo "==> [4/6] Installing PyTorch with CUDA 11.8 support..."
# This overwrites the plain 'torch' from the yaml with a CUDA-aware build.
# Change cu118 → cu121 if your instance has CUDA 12.x
pip install --upgrade \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Pin huggingface_hub — diffusers==0.11.1 is incompatible with huggingface_hub>=0.23
# (cached_download was removed in newer versions)
pip install "huggingface_hub==0.12.1"

# ─── 5. Install diffusion_policy (required by NoMaD) ─────────────────────────
echo ""
echo "==> [5/6] Installing diffusion_policy..."
DIFFPOL_DIR="$REPO_ROOT/diffusion_policy"
if [ ! -d "$DIFFPOL_DIR" ]; then
    git clone https://github.com/real-stanford/diffusion_policy.git "$DIFFPOL_DIR"
else
    echo "     diffusion_policy already cloned, pulling latest..."
    git -C "$DIFFPOL_DIR" pull
fi
pip install -e "$DIFFPOL_DIR"

# ─── 6. Install vint_train package ───────────────────────────────────────────
echo ""
echo "==> [6/6] Installing vint_train package..."
pip install -e "$REPO_ROOT/train/"

# ─── Post-install check ───────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation..."
python - <<'EOF'
import torch
print(f"    torch          : {torch.__version__}")
print(f"    CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU            : {torch.cuda.get_device_name(0)}")
import diffusers; print(f"    diffusers      : {diffusers.__version__}")
import efficientnet_pytorch; print(f"    efficientnet   : OK")
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
print(f"    ConditionalUnet1D : OK")
from vint_train.models.nomad.nomad import NoMaD
print(f"    NoMaD model    : OK")
EOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. conda activate nomad_train"
echo "  2. Place your dataset in the repo and run:"
echo "       cd train && python process_bags.py --help"
echo "  3. Edit train/config/nomad_finetune.yaml — set your data paths"
echo "  4. Register dataset in train/vint_train/data/data_config.yaml"
echo "  5. (Optional) Download pretrained weights from Google Drive:"
echo "       https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg"
echo "     and save as: train/logs/nomad_pretrained/nomad/latest.pth"
echo "     then uncomment 'load_run' in nomad_finetune.yaml"
echo "  6. cd train && python train.py -c config/nomad_finetune.yaml"
echo "============================================================"
