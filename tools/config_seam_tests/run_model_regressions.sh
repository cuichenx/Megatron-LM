#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Run separately inside an existing eight-GPU allocation/container on each revision.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 CHECKOUT NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi

checkout=$(realpath "$1")
output=$(realpath -m "$2")
selector=tests/unit_tests/models/test_hybrid_moe_model.py::TestHybridMoEModel

test -f "$checkout/tests/unit_tests/models/test_hybrid_moe_model.py"
git -C "$checkout" diff --quiet
git -C "$checkout" diff --cached --quiet
# Fail on an existing output directory; never overwrite a prior validation record.
mkdir "$output"

cd "$checkout"
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

git rev-parse HEAD > "$output/revision.txt"
git rev-parse 'HEAD^{tree}' > "$output/tree.txt"
sha256sum tests/unit_tests/models/test_hybrid_moe_model.py > "$output/test-sha256.txt"
uv run --no-sync python -c '
import logging
import torch
logging.basicConfig(level=logging.INFO)
logging.info("torch=%s CUDA=%s GPUs=%s", torch.__version__, torch.version.cuda, torch.cuda.device_count())
if torch.cuda.device_count() != 8:
    raise SystemExit("Expose exactly 8 GPUs: this fixture uses TP=2 and EP=4.")
' > "$output/environment.log" 2>&1

# Preserve failures, including config-golden drift. No xfail, field exclusion,
# test modification, or fixture downsizing is applied by this harness.
set +e
timeout 1200s uv run --no-sync python -m torch.distributed.run \
    --standalone --nproc-per-node=8 --log-dir="$output/ranks" --redirects=3 \
    -m pytest -q --capture=fd "$selector" > "$output/launcher.log" 2>&1
status=$?
set -e
echo "$status" > "$output/exit-status.txt"
exit "$status"
