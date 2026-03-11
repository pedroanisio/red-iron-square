#!/usr/bin/env bash
# DISCLAIMER: No information within should be taken for granted.
# Any statement or premise not backed by a real logical definition
# or verifiable reference may be invalid, erroneous, or a hallucination.

set -euo pipefail

cd backend
rm -rf .mypy_cache
export MYPY_CACHE_DIR=/tmp/red-iron-square-mypy-cache
set +e
.venv/bin/python -m mypy --cache-dir="$MYPY_CACHE_DIR" --no-incremental src/
status=$?
set -e

.venv/bin/python - <<'PY'
from pathlib import Path
import shutil

cache_dir = Path(".mypy_cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
PY

rm -rf "$MYPY_CACHE_DIR"

exit "$status"
