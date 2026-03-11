#!/usr/bin/env bash
# Gate: no source file may exceed 300 LOC (10% tolerance = 330).
# Comments and blank lines are excluded from the count.
set -euo pipefail

LIMIT=330
failed=0

for f in $(find backend/src -name '*.py' ! -name '__init__.py'); do
    loc=$(grep -cv '^\s*#\|^\s*$\|"""' "$f" || true)
    if [ "$loc" -gt "$LIMIT" ]; then
        echo "FAIL: $f has $loc LOC (limit: 300 + 10% = $LIMIT)"
        failed=1
    fi
done

if [ "$failed" -eq 1 ]; then
    exit 1
fi
echo "All source files within 300 LOC limit."
