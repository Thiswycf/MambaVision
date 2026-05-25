#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mambavision"
PY_VER="3.10"

if ! conda env list | rg -q "^${ENV_NAME}\\s"; then
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n "${ENV_NAME}" pip install -r requirements.txt

echo "Environment ready: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}"
