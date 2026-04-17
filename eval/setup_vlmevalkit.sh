#!/bin/bash
# Setup VLMEvalKit for nanoVLM evaluation.
# This script initializes the VLMEvalKit submodule, installs it, and registers
# the nanoVLM adapter. Idempotent: safe to run multiple times.
#
# Usage:
#   bash eval/setup_vlmevalkit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLMEVALKIT_DIR="${REPO_ROOT}/VLMEvalKit"

echo "=== nanoVLM VLMEvalKit Setup ==="
echo "Repo root: ${REPO_ROOT}"

# Step 1: Initialize the VLMEvalKit submodule
echo "[1/4] Initializing VLMEvalKit submodule..."
cd "${REPO_ROOT}"
git submodule update --init --recursive VLMEvalKit

# Step 2: Install VLMEvalKit
echo "[2/4] Installing VLMEvalKit (editable)..."
pip install -e "${VLMEVALKIT_DIR}"

# Step 3: Copy adapter
echo "[3/4] Copying nanoVLM adapter..."
cp "${REPO_ROOT}/eval/vlmevalkit_adapter.py" "${VLMEVALKIT_DIR}/vlmeval/vlm/nanovlm.py"
echo "  -> ${VLMEVALKIT_DIR}/vlmeval/vlm/nanovlm.py"

# Step 4: Patch VLMEvalKit to register nanoVLM
echo "[4/4] Patching VLMEvalKit config..."
python3 - "${VLMEVALKIT_DIR}" <<'PYEOF'
import sys, os

vlmevalkit_dir = sys.argv[1]

# --- Patch __init__.py ---
init_path = os.path.join(vlmevalkit_dir, 'vlmeval', 'vlm', '__init__.py')
with open(init_path, 'r') as f:
    init_content = f.read()

import_line = 'from .nanovlm import NanoVLM'
if import_line not in init_content:
    lines = init_content.rstrip('\n').split('\n')
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('from .'):
            last_import_idx = i
    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, import_line)
    else:
        lines.append(import_line)
    with open(init_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  -> Patched {init_path}")
else:
    print(f"  -> {init_path} already patched, skipping.")

# --- Patch config.py ---
config_path = os.path.join(vlmevalkit_dir, 'vlmeval', 'config.py')
with open(config_path, 'r') as f:
    config_content = f.read()

if 'nanovlm_series' not in config_content:
    nanovlm_block = '''
nanovlm_series = {
    "nanoVLM-460M-8k": partial(vlm.NanoVLM, model_path="lusxvr/nanoVLM-460M-8k"),
    "nanoVLM-230M-8k": partial(vlm.NanoVLM, model_path="lusxvr/nanoVLM-230M-8k"),
}

'''
    register_line = 'model_groups.append(nanovlm_series)\n'
    marker = 'for grp in model_groups:'
    if marker in config_content:
        config_content = config_content.replace(
            marker,
            nanovlm_block + register_line + '\n' + marker
        )
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"  -> Patched {config_path}")
    else:
        print(f"  ERROR: Could not find '{marker}' in {config_path}")
        print(f"  You may need to manually register nanoVLM in config.py")
        sys.exit(1)
else:
    print(f"  -> {config_path} already patched, skipping.")

print("\n=== Setup complete! ===")
print("Run evaluation with: bash eval/eval_vlmevalkit.sh")
PYEOF
