#!/usr/bin/env bash
# DISCLAIMER: No information within should be taken for granted.
# Any statement or premise not backed by a real logical definition
# or verifiable reference may be invalid, erroneous, or a hallucination.

set -euo pipefail

cd backend
rm -f .coverage
rm -rf htmlcov
rm -rf .pytest_cache
export COVERAGE_FILE=/tmp/red-iron-square-coverage
set +e
.venv/bin/python -m pytest tests/ --cov=src --cov-fail-under=80 -q -p no:cacheprovider
status=$?
set -e
rm -f "$COVERAGE_FILE"
rm -rf htmlcov
rm -rf .pytest_cache
exit "$status"
