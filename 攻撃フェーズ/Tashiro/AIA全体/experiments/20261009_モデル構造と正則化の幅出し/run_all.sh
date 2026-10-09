#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="${1:-$(cd ../../../../.. && pwd)}"
python cache_features.py --root "$PROJECT_ROOT"
python model_sweep.py
python model_extra.py
python nested_selection.py
python nested_baselines.py
python outer_extensions.py
python posthoc_analysis.py
python limited_change_models.py
python test_model_width.py
