#!/bin/bash
# ============================================================
# Linux GPU Cloud Environment Setup Script
# Usage: bash setup_linux.sh
# ============================================================
set -e

ENV_NAME="dust3r-gsplat"

echo "[0/7] Cleaning old environment and cache..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Found existing environment ${ENV_NAME}, removing..."
    conda env remove -n ${ENV_NAME} -y
fi
echo "Cleaning conda cache..."
conda clean --all -y
echo "Cleaning pip cache..."
rm -rf ~/.cache/pip/
echo "[0/7] Done."

echo "[1/7] Creating conda environment (Python 3.10)..."
conda create -n dust3r-gsplat python=3.10 -y

echo "[2/7] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dust3r-gsplat

echo "[3/7] Installing PyTorch..."
# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Detected CUDA version: $CUDA_VERSION"

# Determine PyTorch index URL based on CUDA version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

if [ "$CUDA_MAJOR" -eq 11 ]; then
    echo "Using PyTorch for CUDA 11.8..."
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
    PYTORCH_VER="2.4.0"
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    if [ "$CUDA_MINOR" -le 1 ]; then
        echo "Using PyTorch for CUDA 12.1..."
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
        PYTORCH_VER="2.5.0"
    elif [ "$CUDA_MINOR" -le 4 ]; then
        echo "Using PyTorch for CUDA 12.4..."
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
        PYTORCH_VER="2.5.0"
    else
        echo "Using PyTorch 2.8.0 for CUDA 12.5+ (supports sm_120/RTX 50 series)..."
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
        PYTORCH_VER="2.8.0"
    fi
elif [ "$CUDA_MAJOR" -ge 13 ]; then
    echo "Using PyTorch 2.8.0 for CUDA 13.x (supports sm_120/RTX 50 series)..."
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
    PYTORCH_VER="2.8.0"
else
    echo "Unknown CUDA version, using default PyTorch..."
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    PYTORCH_VER="2.4.0"
fi

pip install torch==${PYTORCH_VER} torchvision --index-url $PYTORCH_INDEX

# Verify PyTorch CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[WARN] CUDA not available in PyTorch! Check installation."
fi

echo "[4/7] Installing dependencies..."
pip install -r requirements.txt

echo "[5/7] Installing gsplat..."
# Detect GPU architecture for gsplat compilation
echo "Detecting GPU architecture..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# Map GPU to compute capability and set parallel jobs
# RTX 50 series requires sm_120 - no prebuilt wheel available, must compile from source
if echo "$GPU_NAME" | grep -qi "RTX 50"; then
    GPU_ARCH="12.0"
    MAX_JOBS=1
    NEED_SOURCE_BUILD=true
    echo "Detected RTX 50 series (sm_120) - No prebuilt wheel, compiling from source"
elif echo "$GPU_NAME" | grep -qi "RTX 40"; then
    GPU_ARCH="8.9"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected RTX 40 series (sm_89) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "RTX 30"; then
    GPU_ARCH="8.6"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected RTX 30 series (sm_86) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "RTX 20"; then
    GPU_ARCH="7.5"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected RTX 20 series (sm_75) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "A100"; then
    GPU_ARCH="8.0"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected A100 (sm_80) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "H100"; then
    GPU_ARCH="9.0"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected H100 (sm_90) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "V100"; then
    GPU_ARCH="7.0"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected V100 (sm_70) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "T4"; then
    GPU_ARCH="7.5"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected T4 (sm_75) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "L4\|L40"; then
    GPU_ARCH="8.9"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected L4/L40 (sm_89) - Prebuilt wheel likely available"
elif echo "$GPU_NAME" | grep -qi "A10\|A30"; then
    GPU_ARCH="8.6"
    MAX_JOBS=4
    NEED_SOURCE_BUILD=false
    echo "Detected A10/A30 (sm_86/80) - Prebuilt wheel likely available"
else
    GPU_ARCH="8.6;8.9;9.0"
    MAX_JOBS=2
    NEED_SOURCE_BUILD=true
    echo "Unknown GPU, will compile from source with common architectures"
fi

echo "Using GPU architecture: $GPU_ARCH"

# Try prebuilt wheel first for known GPUs (faster installation)
if [ "$NEED_SOURCE_BUILD" = "false" ]; then
    echo "Attempting to install prebuilt gsplat wheel..."
    if pip install gsplat; then
        echo "Prebuilt wheel installed successfully!"
        python -c "import gsplat; print('gsplat version:', gsplat.__version__)"
    else
        echo "Prebuilt wheel failed, falling back to source build..."
        NEED_SOURCE_BUILD=true
    fi
fi

# Fall back to source build for RTX 50 series or if prebuilt failed
if [ "$NEED_SOURCE_BUILD" = "true" ]; then
    echo "Building gsplat from source (this may take 5-15 minutes)..."
    echo "Using MAX_JOBS=$MAX_JOBS for compilation"
    MAX_JOBS=$MAX_JOBS TORCH_CUDA_ARCH_LIST="$GPU_ARCH" pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
fi

# Verify gsplat installation
python -c "import gsplat; from gsplat import rasterization; print('gsplat imported successfully')"

echo "[6/7] Installing pycolmap..."
pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e --no-build-isolation

echo "[7/7] Installing dust3r..."
if [ ! -d "dust3r" ]; then
    git clone --recursive https://github.com/naver/dust3r
fi

# Fix torch.load weights_only issue for PyTorch 2.6+
echo "Fixing torch.load weights_only compatibility..."
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' dust3r/dust3r/model.py
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' dust3r/dust3r/training.py
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' dust3r/croco/utils/misc.py
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' dust3r/croco/stereoflow/train.py
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' dust3r/croco/stereoflow/test.py
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' dust3r/croco/demo.py

cd dust3r/croco/models/curope
# Use same GPU architecture for dust3r curope extension
TORCH_CUDA_ARCH_LIST="$GPU_ARCH" python setup.py build_ext --inplace
cd ../../..
echo "dust3r ready (auto-loaded via sys.path in script)"

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import gsplat; from gsplat import rasterization; print('gsplat OK')"
python -c "import dust3r; print('dust3r OK')"
python -c "import pycolmap; print('pycolmap OK')"
python -c "import fused_ssim; print('fused_ssim OK (optional)')" 2>/dev/null || echo "fused_ssim not installed (using torchmetrics fallback)"

echo ""
echo "============================================================"
echo "[SUCCESS] Environment ready!"
echo "============================================================"
echo ""
echo "GPU Architecture: $GPU_ARCH (sm_$(echo $GPU_ARCH | tr -d '.'))"
echo ""
echo "Usage:"
echo "  conda activate dust3r-gsplat"
echo "  python dust3r_to_3dgs_verified.py -i ./images -o ./data/raw --save_depth --evaluate"
echo "  python simple_trainer_prune_v2.py --data_dir ./data/raw --dense_depth_dir ./data/raw/depths --enable_prune"
echo ""
echo "Optional: Install fused-ssim for faster SSIM:"
echo "  MAX_JOBS=1 pip install git+https://github.com/rahul-goel/fused-ssim --no-build-isolation"