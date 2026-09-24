"""Zhang11 C의 공개 희소행 6개를 A_bg 희소행 쪽으로 완화한다.

이 코드는 사후 가공 실험의 재현용이다. 차등프라이버시 보증이나
본선의 공식 MIA/AIA 점수 향상을 주장하지 않는다. 원본 ZIP과 토큰은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CONTINUOUS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
LOG_COLUMNS = frozenset(("TG", "ALT"))
N_ROWS_TO_SMOOTH = 6
BACKGROUND_WEIGHT = 0.90
MIN_LOCAL_UTILITY = 0.90


def transformed(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, CONTINUOUS].to_numpy(dtype=float).copy()
    for column_index, name in enumerate(CONTINUOUS):
        if name in LOG_COLUMNS:
            values[:, column_index] = np.log1p(values[:, column_index])
    if not np.isfinite(values).all():
        raise ValueError("연속 속성에 결측값, 무한대 또는 잘못된 로그 입력값이 있습니다.")
    return values


def load_original_c(source_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(source_zip) as archive:
        if "C.csv" not in archive.namelist():
            raise ValueError("원본 ZIP 바로 아래에 C.csv가 있어야 합니다.")
        return pd.read_csv(io.BytesIO(archive.read("C.csv")), float_precision="round_trip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True, help="본선 스타터 키트 폴더")
    parser.add_argument("--zhang11-zip", type=Path, required=True, help="원본 Zhang11 ZIP")
    parser.add_argument("--out-c", type=Path, required=True, help="새 C.csv 저장 경로; 기존 파일은 덮어쓰지 않음")
    args = parser.parse_args()

    kit_src = args.kit / "starter" / "src"
    dist = args.kit / "participant_data"
    if not kit_src.is_dir() or not dist.is_dir():
        parser.error("--kit에 본선 키트 최상위 폴더를 지정하세요.")
    if not args.zhang11_zip.is_file():
        parser.error("--zhang11-zip 파일을 찾을 수 없습니다.")
    if args.out_c.suffix.lower() != ".csv":
        parser.error("--out-c는 .csv 파일이어야 합니다.")
    if args.out_c.exists():
        parser.error("출력 파일이 이미 있습니다. 기존 파일을 덮어쓰지 않습니다.")

    sys.path.insert(0, str(kit_src))
    from pwscup2026.common.submission_schema import validate_c_submission
    from pwscup2026.kit.selfscore import load_local_reference, self_score
    from pwscup2026.rare.outlier import mahalanobis

    schema = json.loads((dist / "schema.json").read_text(encoding="utf-8"))
    detector_cfg = schema["rare_detector"]
    ranges = schema["ranges"]
    mean = np.asarray(detector_cfg["mu"], dtype=float)
    covariance = np.asarray(detector_cfg["sigma"], dtype=float)
    threshold = float(detector_cfg["band"]["lo"])
    detector_columns = detector_cfg["columns"]

    def is_public_rare(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            mahalanobis(frame, mean, covariance, detector_columns) >= threshold,
            dtype=bool,
        )

    a_bg = pd.read_csv(dist / "A_bg.csv", float_precision="round_trip")
    b, reference, score_cfg, _ = load_local_reference(dist)
    b_self = pd.read_csv(dist / "B_self.csv", float_precision="round_trip")
    original = load_original_c(args.zhang11_zip)

    if list(original.columns) != [name for name in b.columns if name != "record_id"]:
        raise ValueError("Zhang11 C의 열·열 순서가 본선 B와 다릅니다.")
    if not b["record_id"].equals(b_self["record_id"]):
        raise ValueError("B.csv와 B_self.csv의 record_id 순서가 다릅니다.")
    if b_self["is_rare"].isna().any() or not set(b_self["is_rare"].unique()).issubset({0, 1, False, True}):
        raise ValueError("B_self.csv의 is_rare 열은 결측 없는 0/1이어야 합니다.")
    true_rare_b = b_self["is_rare"].to_numpy(dtype=bool)
    original_check = validate_c_submission(original, score_cfg, n_expected=len(b))
    if not original_check.ok:
        raise ValueError("원본 C의 제출 형식이 유효하지 않습니다: " + "; ".join(original_check.errors))

    rare_c_indices = np.flatnonzero(is_public_rare(original))
    a_rare = a_bg.loc[is_public_rare(a_bg)]
    if len(rare_c_indices) < N_ROWS_TO_SMOOTH or a_rare.empty or not true_rare_b.any():
        raise ValueError("희소행 또는 A_bg 희소 기증행이 부족합니다.")

    # 1. B의 참 희소인원과 가까운 C 공개 희소행 6개를 찾는다.
    spread = np.std(transformed(a_bg), axis=0, ddof=0)
    spread = np.where(spread > 0, spread, 1.0)
    c_rare_scaled = transformed(original.loc[rare_c_indices]) / spread
    b_rare_scaled = transformed(b.loc[true_rare_b]) / spread
    distance_to_b = cKDTree(b_rare_scaled).query(c_rare_scaled, k=1)[0]
    chosen = rare_c_indices[np.argsort(distance_to_b, kind="stable")[:N_ROWS_TO_SMOOTH]]

    # 2. 각 선택행을 가장 가까운 A_bg 공개 희소행 쪽으로 90% 이동한다.
    a_rare_transformed = transformed(a_rare)
    c_chosen_transformed = transformed(original.loc[chosen])
    donor_positions = cKDTree(a_rare_transformed / spread).query(
        c_chosen_transformed / spread, k=1
    )[1]
    donor = a_rare_transformed[donor_positions]
    blended = (1.0 - BACKGROUND_WEIGHT) * c_chosen_transformed + BACKGROUND_WEIGHT * donor
    candidate = original.copy(deep=True)
    for j, name in enumerate(CONTINUOUS):
        value = np.expm1(blended[:, j]) if name in LOG_COLUMNS else blended[:, j]
        if name == "age":
            value = np.rint(np.clip(value, 40, 74)).astype(int)
        else:
            value = np.clip(value, *ranges[name])
        candidate.loc[chosen, name] = value

    changed_rows = int((candidate != original).any(axis=1).sum())
    unchanged_columns = [name for name in original.columns if name not in CONTINUOUS]
    if changed_rows != N_ROWS_TO_SMOOTH:
        raise ValueError(f"예상 변경 행 수 6과 다릅니다: {changed_rows}")
    if not candidate[unchanged_columns].equals(original[unchanged_columns]):
        raise ValueError("범주형 속성이나 time/onset/death가 변경됐습니다.")
    if not candidate.drop(index=chosen).equals(original.drop(index=chosen)):
        raise ValueError("선택하지 않은 행이 변경됐습니다.")
    if int(is_public_rare(candidate).sum()) != len(rare_c_indices):
        raise ValueError("C의 공개 희소행 수가 바뀌었습니다.")

    # 3. 실제 저장될 CSV 문자열을 다시 읽어 형식과 유용성을 확인한다.
    buffer = io.StringIO()
    candidate.to_csv(buffer, index=False)
    csv_text = buffer.getvalue()
    reloaded = pd.read_csv(io.StringIO(csv_text), float_precision="round_trip")
    submission_check = validate_c_submission(reloaded, score_cfg, n_expected=len(b))
    if not submission_check.ok:
        raise ValueError("출력 CSV의 제출 형식이 유효하지 않습니다: " + "; ".join(submission_check.errors))
    baseline_score = self_score(original, b, reference, score_cfg)
    candidate_score = self_score(reloaded, b, reference, score_cfg)
    utility = float(candidate_score["utility"])
    if not np.isfinite(utility) or utility < MIN_LOCAL_UTILITY:
        raise ValueError(f"로컬 유용성 {utility:.6f}이 최소 조건 0.9에 미달하여 저장하지 않습니다.")

    args.out_c.parent.mkdir(parents=True, exist_ok=True)
    with args.out_c.open("x", encoding="utf-8", newline="") as output:
        output.write(csv_text)
    print(f"저장: {args.out_c}")
    print(f"변경: C의 {len(rare_c_indices)}개 공개 희소행 중 {changed_rows}행, 연속 7속성만 변경")
    print(f"원본 로컬 U: {float(baseline_score['utility']):.6f}")
    print(f"완화안 로컬 U: {utility:.6f}")
    print("주의: 이 수치는 본선 키트의 로컬 유용성이지 공식 MIA/AIA 보호 점수가 아닙니다.")


if __name__ == "__main__":
    main()
