#!/bin/bash
# ============================================================
# FCBG Interview Demo — Environment Setup (venv)
# Usage: bash setup_env_venv.sh
# ============================================================

set -e

echo "============================================"
echo "  FCBG Interview Demo — Environment Setup"
echo "============================================"
echo ""

# --- 1. Create virtual environment ---
echo "[1/4] Creating virtual environment 'fcbg-env'..."
python3.10 -m venv fcbg-env
source fcbg-env/bin/activate
pip install --upgrade pip
echo ""

# --- 2. Install PyTorch with CUDA support ---
echo "[2/4] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo ""

# --- 3. Install core packages ---
echo "[3/4] Installing core packages..."
pip install -r requirements.txt
echo ""

# --- 4. Install DeepLabCut and LISBET ---
echo "[4/4] Installing DeepLabCut and LISBET..."
pip install "deeplabcut[gui,tf]"
pip install lisbet
echo ""

# --- Create project directory structure ---
mkdir -p videos output notebooks figures scripts

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. source fcbg-env/bin/activate"
echo "  2. Place your videos in videos/"
echo "  3. jupyter notebook notebooks/FCBG_CV_Demo.ipynb"
echo ""
