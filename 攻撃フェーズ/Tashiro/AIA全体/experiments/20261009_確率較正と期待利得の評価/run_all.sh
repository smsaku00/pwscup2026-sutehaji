#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -ne 3 ]; then
  echo 'Usage: ./run_all.sh /path/to/repo /path/to/generated_density_outputs /path/to/new_calibration_outputs' >&2
  exit 2
fi
REPO="$(cd "$1" && pwd)"
DENS="$(cd "$2" && pwd)"
PRIOR="${AIA_PRIOR:-$(cd ../20261009_標的対照の独立証拠とソフト群制約 && pwd)}"
OUT="$3"
if [ "$OUT" = '.' ] || [ "$(realpath -m "$OUT")" = "$(pwd)" ]; then
  echo 'Refusing to overwrite the source experiment directory' >&2
  exit 3
fi
for f in "$DENS/density_public_scores.csv" "$DENS/augment_public_predictions.csv" "$DENS/augment_target_metrics.csv" "$DENS/per_team_metrics.csv" "$PRIOR/experiment.py"; do
  if [ ! -f "$f" ]; then echo "Missing prerequisite: $f" >&2; exit 4; fi
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for FOLD in 3 4 5; do
  DEST="$OUT"
  if [ "$FOLD" != 4 ]; then DEST="$OUT/fold$FOLD"; fi
  python calibrated_policy.py --root "$REPO" --prior "$PRIOR" --density-scores "$DENS/density_public_scores.csv" --outer-scores "$DENS/augment_public_predictions.csv" --out "$DEST" --folds "$FOLD" > "$OUT/fold${FOLD}_run.log"
  if [ "$FOLD" != 4 ]; then cp "$DEST/summary.csv" "$OUT/fold${FOLD}_summary.csv"; fi
done
python processed_pool.py --root "$REPO" --prior "$PRIOR" --density-scores "$DENS/density_public_scores.csv" --out "$OUT/processed_pool.csv" > "$OUT/processed_pool.log"
python comparison.py --prior-density "$DENS/per_team_metrics.csv" --augmentation "$DENS/augment_target_metrics.csv" --calibration "$OUT/per_target.csv" --out "$OUT" > "$OUT/comparison.log"
AIA_ROOT="$REPO" AIA_PRIOR="$PRIOR" python -m unittest -v test_calibrated_policy > "$OUT/tests.log" 2>&1
printf 'All analyses and 8 tests completed: %s\n' "$OUT"
