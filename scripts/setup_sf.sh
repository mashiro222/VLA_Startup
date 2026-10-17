#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/sf_common.sh"
rev=372819b31ba9a2c8fc5989edc7d525cb187cecd5
if [[ ! -d "$SF_REPO/.git" ]]; then
    git clone https://github.com/OpenHelix-Team/Spatial-Forcing.git "$SF_REPO"
    git -C "$SF_REPO" checkout --detach "$rev"
fi
[[ "$(git -C "$SF_REPO" rev-parse HEAD)" == "$rev" ]]
[[ -z "$(git -C "$SF_REPO" status --porcelain)" ]]
if [[ ! -x "$SF_ENV/bin/python" ]]; then "$UV" venv --python 3.11 "$SF_ENV"; fi
"$UV" pip sync --python "$SF_ENV/bin/python" requirements/sf_train.lock
# Patch only this isolated environment; existing evaluation environments are unchanged.
"$SF_ENV/bin/python" - <<'PY'
import os,pathlib,shutil,transformers
src=pathlib.Path(os.environ['SF_REPO'])/'openpi-SF/src/openpi/models_pytorch/transformers_replace'
dst=pathlib.Path(transformers.__file__).parent
shutil.copytree(src,dst,dirs_exist_ok=True)
PY
"$SF_ENV/bin/python" -c 'from openpi.models_pytorch.pi0_align_pytorch import PI0Pytorch; from vggt.models.vggt import VGGT; print("SF training imports OK")'
