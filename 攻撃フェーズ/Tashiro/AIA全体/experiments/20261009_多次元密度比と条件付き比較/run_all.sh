#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ $# -lt 1 ]; then
  echo 'Usage: ./run_all.sh /path/to/pwscup2026-sutehaji [/path/to/results]' >&2
  exit 2
fi
ROOT="$1"; OUT="${2:-./results}"
PRIOR="$(dirname "$PWD")/20261009_標的対照の独立証拠とソフト群制約"
if [ ! -f "$PRIOR/experiment.py" ]; then
  echo "Earlier baseline experiment.py not found: $PRIOR" >&2; exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python prepare_prior.py --root "$ROOT" --prior "$PRIOR" --out "$OUT/prior_public_scores.csv"
python experiment.py --root "$ROOT" --out "$OUT"
python negative_controls.py --root "$ROOT" --out "$OUT"
python background_sensitivity.py --root "$ROOT" --out "$OUT"
python augment_existing.py --root "$ROOT" --prior "$PRIOR" --out "$OUT"
python -m unittest -v test_experiment
