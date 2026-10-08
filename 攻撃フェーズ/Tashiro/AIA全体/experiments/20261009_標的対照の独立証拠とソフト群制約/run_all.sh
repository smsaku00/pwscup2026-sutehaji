#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="${1:?Usage: ./run_all.sh /path/to/repo-root [out-dir]}"
OUT="${2:-./output}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiment.py --root "$ROOT" --out "$OUT"
python extension.py --root "$ROOT" --scores "$OUT/public_scores.csv" --out "$OUT"
python negative_controls.py --root "$ROOT" --out "$OUT"
python diagnostics.py --root "$ROOT" --out "$OUT"
python pairwise_training.py --root "$ROOT" --out "$OUT"
