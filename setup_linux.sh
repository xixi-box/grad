#!/bin/bash
# ============================================================
# Linux GPU Environment Setup Script for dust3r-gsplat
# Usage: bash setup_linux.sh
# Must be run from project root (where this script is located)
# ============================================================

set -euo pipefail

ENV_NAME="dust3r-gsplat"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_GSPLAT_DIR="${ROOT_DIR}/gsplat"
LOCAL_DUST3R_DIR="${ROOT_DIR}/dust3r"

echo "============================================================"
echo "Project root : ${ROOT_DIR}"
echo "Environment  : ${ENV_NAME}"
echo "============================================================"

# ------------------------------------------------------------
# 0. Check conda
# ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# ------------------------------------------------------------
# 1. Create or activate conda env
# ------------------------------------------------------------
echo "[1/8] Preparing conda environment..."
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Found existing environment: ${ENV_NAME}"
else
    echo "Creating environment: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"
echo "[1/8] Done."

# ------------------------------------------------------------
# 2. Detect GPU / CUDA
# ------------------------------------------------------------
echo "[2/8] Detecting GPU and CUDA..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. NVIDIA driver may not be installed."
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/[[:space:]]*$//')"
CUDA_VERSION="$(nvidia-smi | grep 'CUDA Version' | awk '{print $9}' | head -1 || true)"

if [ -z "${GPU_NAME}" ]; then
    echo "[ERROR] Failed to detect GPU name."
    exit 1
fi

if [ -z "${CUDA_VERSION}" ]; then
    echo "[WARN] Failed to detect CUDA version from nvidia-smi, defaulting to 12.1 strategy."
    CUDA_VERSION="12.1"
fi

echo "Detected GPU : ${GPU_NAME}"
echo "Detected CUDA: ${CUDA_VERSION}"

CUDA_MAJOR="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
CUDA_MINOR="$(echo "${CUDA_VERSION}" | cut -d. -f2)"

IS_50_SERIES=false
GPU_ARCH=""
MAX_JOBS=4

if echo "${GPU_NAME}" | grep -Eqi 'RTX 50|RTX 5060|RTX 5070|RTX 5080|RTX 5090'; then
    IS_50_SERIES=true
    GPU_ARCH="12.0"
    MAX_JOBS=1
    echo "GPU class    : RTX 50 series"
elif echo "${GPU_NAME}" | grep -Eqi 'RTX 40|RTX 4090|RTX 4080|RTX 4070|RTX 4060|L4|L40'; then
    GPU_ARCH="8.9"
    MAX_JOBS=4
    echo "GPU class    : RTX 40 series / Ada"
elif echo "${GPU_NAME}" | grep -Eqi 'RTX 30|RTX 3090|RTX 3080|RTX 3070|RTX 3060|A10|A30'; then
    GPU_ARCH="8.6"
    MAX_JOBS=4
    echo "GPU class    : RTX 30 series / Ampere"
elif echo "${GPU_NAME}" | grep -Eqi 'A100'; then
    GPU_ARCH="8.0"
    MAX_JOBS=4
    echo "GPU class    : A100"
elif echo "${GPU_NAME}" | grep -Eqi 'H100'; then
    GPU_ARCH="9.0"
    MAX_JOBS=4
    echo "GPU class    : H100"
elif echo "${GPU_NAME}" | grep -Eqi 'V100'; then
    GPU_ARCH="7.0"
    MAX_JOBS=4
    echo "GPU class    : V100"
elif echo "${GPU_NAME}" | grep -Eqi 'T4|RTX 20|RTX 2080|RTX 2070|RTX 2060'; then
    GPU_ARCH="7.5"
    MAX_JOBS=4
    echo "GPU class    : Turing"
else
    GPU_ARCH=""
    MAX_JOBS=2
    echo "GPU class    : Unknown, will use conservative fallback"
fi

echo "TORCH_CUDA_ARCH_LIST target: ${GPU_ARCH:-auto}"

# ------------------------------------------------------------
# 3. Install PyTorch
# ------------------------------------------------------------
echo "[3/8] Installing PyTorch..."

PYTORCH_INDEX=""
PYTORCH_VER=""

if [ "${CUDA_MAJOR}" -eq 11 ]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
    PYTORCH_VER="2.4.0"
elif [ "${CUDA_MAJOR}" -eq 12 ]; then
    if [ "${IS_50_SERIES}" = true ]; then
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
        PYTORCH_VER="2.8.0"
    else
        if [ "${CUDA_MINOR}" -le 1 ]; then
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
            PYTORCH_VER="2.5.0"
        elif [ "${CUDA_MINOR}" -le 4 ]; then
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
            PYTORCH_VER="2.5.0"
        else
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
            PYTORCH_VER="2.8.0"
        fi
    fi
else
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu129"
    PYTORCH_VER="2.8.0"
fi

echo "Installing torch==${PYTORCH_VER} from ${PYTORCH_INDEX}"
pip install --upgrade pip
pip install "torch==${PYTORCH_VER}" torchvision torchaudio --index-url "${PYTORCH_INDEX}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# 4. Install dependencies
# ------------------------------------------------------------
echo "[4/8] Installing dependencies..."
if [ -f "${ROOT_DIR}/requirements.txt" ]; then
    pip install -r "${ROOT_DIR}/requirements.txt"
else
    echo "[WARN] requirements.txt not found, skipping."
fi

# ------------------------------------------------------------
# 5. Install gsplat
# ------------------------------------------------------------
echo "[5/8] Installing gsplat..."

if [ "${IS_50_SERIES}" = true ]; then
    echo "50 series detected -> local source strategy."

    if [ ! -d "${LOCAL_GSPLAT_DIR}" ]; then
        echo "Local gsplat not found, cloning..."
        git clone https://github.com/nerfstudio-project/gsplat.git "${LOCAL_GSPLAT_DIR}"
    else
        echo "Found local gsplat: ${LOCAL_GSPLAT_DIR}"
    fi

    pip uninstall -y gsplat || true
    rm -rf ~/.cache/torch_extensions/*

    find "${LOCAL_GSPLAT_DIR}" -type d -name build -exec rm -rf {} + 2>/dev/null || true
    find "${LOCAL_GSPLAT_DIR}" -type d -name dist -exec rm -rf {} + 2>/dev/null || true
    find "${LOCAL_GSPLAT_DIR}" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    if [ -n "${GPU_ARCH}" ]; then
        export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
    fi
    export MAX_JOBS="${MAX_JOBS}"

    echo "Installing local gsplat..."
    pushd "${LOCAL_GSPLAT_DIR}" >/dev/null
    pip install -e . --no-build-isolation
    popd >/dev/null

    if [ -d "${LOCAL_GSPLAT_DIR}/fused-bilagrid" ]; then
        echo "Installing local fused-bilagrid..."
        pushd "${LOCAL_GSPLAT_DIR}/fused-bilagrid" >/dev/null
        pip install -e . --no-build-isolation || true
        popd >/dev/null
    fi
else
    echo "Non-50 series detected -> previous Linux strategy."
    echo "Trying prebuilt gsplat package first..."
    if pip install gsplat; then
        echo "Prebuilt gsplat installed successfully."
    else
        echo "Prebuilt install failed, falling back to source build..."
        TMP_GSPLAT_DIR="${ROOT_DIR}/_tmp_gsplat_build"
        rm -rf "${TMP_GSPLAT_DIR}"
        git clone https://github.com/nerfstudio-project/gsplat.git "${TMP_GSPLAT_DIR}"

        if [ -n "${GPU_ARCH}" ]; then
            export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
        fi
        export MAX_JOBS="${MAX_JOBS}"

        pushd "${TMP_GSPLAT_DIR}" >/dev/null
        pip install -e . --no-build-isolation
        popd >/dev/null
    fi
fi

# ------------------------------------------------------------
# 6. Install pycolmap
# ------------------------------------------------------------
echo "[6/8] Installing pycolmap..."
pip install "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" --no-build-isolation

# ------------------------------------------------------------
# 7. Prepare dust3r
# ------------------------------------------------------------
echo "[7/8] Preparing dust3r..."
if [ ! -d "${LOCAL_DUST3R_DIR}" ]; then
    echo "Local dust3r not found, cloning..."
    git clone --recursive https://github.com/naver/dust3r "${LOCAL_DUST3R_DIR}"
else
    echo "Found local dust3r: ${LOCAL_DUST3R_DIR}"
fi

echo "Patching torch.load(weights_only=False) for newer torch..."
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "${LOCAL_DUST3R_DIR}/dust3r/model.py" || true
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "${LOCAL_DUST3R_DIR}/dust3r/training.py" || true
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "${LOCAL_DUST3R_DIR}/croco/utils/misc.py" || true
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "${LOCAL_DUST3R_DIR}/croco/stereoflow/train.py" || true
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "${LOCAL_DUST3R_DIR}/croco/stereoflow/test.py" || true
sed -i 's/torch.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "${LOCAL_DUST3R_DIR}/croco/demo.py" || true

if [ -d "${LOCAL_DUST3R_DIR}/croco/models/curope" ]; then
    echo "Building dust3r curope extension..."
    pushd "${LOCAL_DUST3R_DIR}/croco/models/curope" >/dev/null
    if [ -n "${GPU_ARCH}" ]; then
        TORCH_CUDA_ARCH_LIST="${GPU_ARCH}" python setup.py build_ext --inplace
    else
        python setup.py build_ext --inplace
    fi
    popd >/dev/null
fi

# ------------------------------------------------------------
# 8. Write conda hooks
# ------------------------------------------------------------
echo "[8/8] Writing conda activate hooks..."
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/env_vars.sh" <<EOF
#!/bin/bash
export PROJECT_ROOT="${ROOT_DIR}"
export GSPLAT_DIR="${LOCAL_GSPLAT_DIR}"
export DUST3R_DIR="${LOCAL_DUST3R_DIR}"
export PYTHONPATH="\${GSPLAT_DIR}:\${DUST3R_DIR}:\${PYTHONPATH}"
EOF

if [ -n "${GPU_ARCH}" ]; then
cat >> "${ACTIVATE_DIR}/env_vars.sh" <<EOF
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
EOF
fi

cat > "${DEACTIVATE_DIR}/env_vars.sh" <<'EOF'
#!/bin/bash
unset PROJECT_ROOT
unset GSPLAT_DIR
unset DUST3R_DIR
unset TORCH_CUDA_ARCH_LIST
EOF

chmod +x "${ACTIVATE_DIR}/env_vars.sh" "${DEACTIVATE_DIR}/env_vars.sh"

# Re-activate current env so hooks take effect immediately
conda deactivate
conda activate "${ENV_NAME}"

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python - <<PY
import os
import gsplat
print("gsplat file:", gsplat.__file__)
from gsplat import rasterization
print("gsplat rasterization: OK")
print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY

python - <<PY
import sys
sys.path.insert(0, "${LOCAL_DUST3R_DIR}")
import dust3r
print("dust3r: OK")
PY

python - <<PY
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "============================================================"
echo "[SUCCESS] Environment ready!"
echo "============================================================"
echo "GPU Name             : ${GPU_NAME}"
echo "CUDA Version         : ${CUDA_VERSION}"
echo "Is 50 Series         : ${IS_50_SERIES}"
echo "TORCH_CUDA_ARCH_LIST : ${GPU_ARCH:-auto}"
echo "Project Root         : ${ROOT_DIR}"
echo ""
echo "Suggested check:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import gsplat; print(gsplat.__file__); from gsplat import rasterization; print('gsplat ok')\""
echo ""
echo "Training example:"
echo "  python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths ..."