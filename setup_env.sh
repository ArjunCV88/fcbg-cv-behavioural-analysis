#!/bin/bash
# ============================================================
# FCBG Interview Demo — Environment Setup
# Run this script from the directory where you want to work
# Usage: bash setup_env.sh
# ============================================================

set -e

echo "============================================"
echo "  FCBG Interview Demo — Environment Setup"
echo "============================================"
echo ""

# --- 1. Create conda environment ---
echo "[1/6] Creating conda environment 'fcbg-demo' with Python 3.10..."
conda create -n fcbg-demo python=3.10 -y
echo ""

# --- 2. Activate environment ---
echo "[2/6] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate fcbg-demo
echo ""

# --- 3. Install PyTorch with CUDA support ---
echo "[3/6] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# --- 4. Install core packages ---
echo "[4/6] Installing core packages (MediaPipe, OpenCV, Jupyter, plotting)..."
pip install \
    mediapipe \
    opencv-python \
    opencv-contrib-python \
    matplotlib \
    numpy \
    pandas \
    scipy \
    jupyter \
    ipywidgets \
    tqdm \
    Pillow \
    seaborn
echo ""

# --- 5. Install DeepLabCut ---
echo "[5/6] Installing DeepLabCut..."
pip install "deeplabcut[gui,tf]"
# Also install DLC with PyTorch engine (DLC 3.x)
pip install "deeplabcut[gui]"
echo ""

# --- 6. Install LISBET ---
echo "[6/6] Installing LISBET..."
pip install lisbet
echo ""

# --- Create project directory structure ---
echo "Creating project directory structure..."
mkdir -p fcbg_demo/{videos,output,notebooks,figures}
echo ""

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. conda activate fcbg-demo"
echo "  2. Place your dancing video in fcbg_demo/videos/"
echo "  3. cd fcbg_demo/notebooks"
echo "  4. jupyter notebook"
echo "  5. Open the demo notebook and run it!"
echo ""
echo "If DeepLabCut install has issues, try:"
echo "  pip install deeplabcut --upgrade"
echo ""
