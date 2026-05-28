#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="humannova"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME"
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.4.1+cu121 \
  torchvision==0.19.1+cu121 \
  torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install ninja cython

TMP_REQUIREMENTS="$(mktemp)"
grep -Ev '^(git\+https://github.com/facebookresearch/detectron2|chumpy)$' "$SCRIPT_DIR/requirements.txt" > "$TMP_REQUIREMENTS"
python -m pip install -r "$TMP_REQUIREMENTS"
rm -f "$TMP_REQUIREMENTS"

python -m pip install flash-attn --no-build-isolation
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
python -m pip install onnxruntime smplx pytorch_lightning pyrender
python -m pip install --no-build-isolation git+https://github.com/mattloper/chumpy
python -m pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2
python -m pip install --no-cache-dir --force-reinstall "setuptools==81.0.0"
python -m pip install --no-cache-dir --force-reinstall "transformers==4.49.0"
python -m pip install --force-reinstall numpy==1.24.4 opencv-python==3.4.18.65 opencv-python-headless==3.4.18.65

cat <<'EOF'

Environment setup complete.

EOF
