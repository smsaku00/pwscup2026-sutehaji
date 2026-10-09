#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -ne 4 ]; then
  echo 'Usage: ./run_all.sh /path/to/repo /path/to/local-density-output /path/to/local-prior-experiment /path/to/local-result-output' >&2
  exit 2
fi
REPO="$1"; DENS="$2"; PRIOR="$3"; OUT="$4"
CAL="../20261009_確率較正と期待利得の評価"
if [ ! -f "$CAL/calibrated_policy.py" ]; then echo 'Missing required prior calibration code' >&2; exit 3; fi
if [ ! -f "$DENS/density_public_scores.csv" ] || [ ! -f "$DENS/augment_public_predictions.csv" ]; then echo 'Run the parent density experiment first' >&2; exit 4; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
case "$OUT" in
  "$(pwd)"|"$(pwd)/"*) echo 'Refusing to overwrite source directory/results. Use an external output directory.' >&2; exit 5;;
esac
export AIA_WIDTH_RESULTS="$OUT" AIA_CAL_DIR="$CAL" AIA_ROOT="$REPO" AIA_PRIOR="$PRIOR"
python width_sweep.py --root "$REPO" --prior "$PRIOR" \
  --density-scores "$DENS/density_public_scores.csv" \
  --outer-scores "$DENS/augment_public_predictions.csv" \
  --calibration-dir "$CAL" --out "$OUT" --all-folds
python summarize.py --results "$OUT"
python compress_release.py --results "$OUT"
python -m unittest -v test_width_sweep
printf 'Completed sweep and tests: %s\n' "$OUT"
