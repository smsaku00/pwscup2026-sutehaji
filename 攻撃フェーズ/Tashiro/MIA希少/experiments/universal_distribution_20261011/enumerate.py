"""Regenerate all 82,995,960 abstract labelled count vectors.

Requires an explicit new output directory. No competition input files or patient
IDs are used. The raw payload is reproducible independent of gzip implementation.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from math import comb
from pathlib import Path
import sys
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "analysis/universal_distribution_20261011"
Q = 22
SLOT_MASK = (1 << Q) - 1
BYTE_POPCOUNT = np.array([v.bit_count() for v in range(256)], dtype=np.uint8)
POWERS = np.left_shift(np.uint32(1), np.arange(Q, dtype=np.uint32))

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while raw := stream.read(8 * 1024 * 1024):
            h.update(raw)
    return h.hexdigest()

def combination_masks(n: int, k: int) -> np.ndarray:
    """Increasing integer masks (colex combinations), independently counted."""
    require(0 <= k <= n <= Q, "Invalid combination domain")
    if k == 0:
        return np.array([0], dtype=np.uint32)
    def masks():
        value = (1 << k) - 1
        limit = 1 << n
        while value < limit:
            yield value
            low = value & -value
            raised = value + low
            value = raised | (((raised ^ value) >> 2) // low)
    result = np.fromiter(masks(), dtype=np.uint32, count=comb(n, k))
    require(len(result) == comb(n, k), "Combination cardinality differs")
    require(len(result) == 1 or bool(np.all(result[1:] > result[:-1])), "Combination order differs")
    return result


def deposited_codes(mask2_batch: np.ndarray, relative_mask1: np.ndarray, remaining_slots: int) -> np.ndarray:
    """Vectorized bit deposit into the complement of each two-member mask."""
    available = np.broadcast_to(POWERS, (len(mask2_batch), Q))[
        (mask2_batch[:, None] & POWERS[None, :]) == 0
    ].reshape(len(mask2_batch), remaining_slots)
    mask1 = np.zeros((len(mask2_batch), len(relative_mask1)), dtype=np.uint32)
    for start in range(0, remaining_slots, 6):
        width = min(6, remaining_slots - start)
        digits = (relative_mask1 >> start) & ((1 << width) - 1)
        binary_digits = ((np.arange(1 << width, dtype=np.uint32)[:, None]
                          >> np.arange(width, dtype=np.uint32)[None, :]) & 1)
        # Distinct power-of-two weights let integer sum implement bitwise OR.
        lookup = (binary_digits @ available[:, start:start + width].T).T
        mask1 |= lookup[:, digits]
    codes = mask1.astype(np.uint64)
    codes |= mask2_batch.astype(np.uint64)[:, None] << Q
    return codes.reshape(-1)


def popcounts_uint32(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.uint32)
    return BYTE_POPCOUNT[contiguous.view(np.uint8).reshape(-1, 4)].sum(axis=1)


def validate_codes(codes: np.ndarray, a: int, b: int, previous: int | None) -> int:
    require(len(codes) > 0, "Empty chunk")
    require(int(codes[-1]) < (1 << (2 * Q)), "Code exceeds 44-bit format")
    require(bool(np.all(codes[1:] > codes[:-1])), "Chunk is not strictly increasing")
    require(previous is None or int(codes[0]) > previous, "Cross-chunk duplicate or order failure")
    mask1 = (codes & SLOT_MASK).astype(np.uint32)
    mask2 = (codes >> Q).astype(np.uint32)
    require(bool(np.all((mask1 & mask2) == 0)), "One/two masks overlap")
    require(bool(np.all(popcounts_uint32(mask1) == a)), "Incorrect one-member target count")
    require(bool(np.all(popcounts_uint32(mask2) == b)), "Incorrect two-member target count")
    return int(codes[-1])


def generate(output_dir: Path, batch_records: int) -> dict:
    require(batch_records >= 1024, "batch-records must be at least 1024")
    require(not output_dir.exists(), "Refusing to overwrite an existing output directory")
    metadata_path = DATA / "fixed5/enumeration_manifest.json"
    expected = json.loads(metadata_path.read_text(encoding="utf-8"))
    ordered_types = [(a, b) for b in range(6) for a in range(12 - 2*b)]
    require([(v["a"], v["b"]) for v in expected["blocks"]] == ordered_types,
            "Metadata does not contain the ordered 42 types")
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = output_dir / "全人数パターン.uint64.gz"
    whole = hashlib.sha256()
    total = exact11 = 0
    blocks = []
    with payload.open("xb") as output:
        for i, ((a, b), ref) in enumerate(zip(ordered_types, expected["blocks"])):
            mask2_values = combination_masks(Q, b)
            relative_mask1 = combination_masks(Q-b, a)
            batch_size = max(1, min(4096, batch_records // len(relative_mask1)))
            block_hash = hashlib.sha256()
            previous = None
            count = 0
            with gzip.GzipFile(filename="", mode="wb", fileobj=output,
                               compresslevel=1, mtime=0) as member:
                for start in range(0, len(mask2_values), batch_size):
                    codes = deposited_codes(mask2_values[start:start+batch_size], relative_mask1, Q-b)
                    previous = validate_codes(codes, a, b, previous)
                    raw = codes.astype("<u8", copy=False).tobytes(order="C")
                    member.write(raw)
                    block_hash.update(raw)
                    whole.update(raw)
                    count += len(codes)
            require(count == comb(Q,b)*comb(Q-b,a) == ref["count"], "Block count mismatch")
            require(block_hash.hexdigest() == ref["raw_sha256"], "Block raw SHA mismatch")
            blocks.append({"a":a, "b":b, "count":count, "raw_sha256":block_hash.hexdigest()})
            total += count
            if a+2*b == 11:
                exact11 += count
            print(f"block {i+1}/42: a={a}, b={b}, records={count:,}", file=sys.stderr, flush=True)
    require(total == expected["total_records"] == 82_995_960, "Total count mismatch")
    require(exact11 == expected["exact11_records"] == 49_402_850, "Exact-11 count mismatch")
    require(whole.hexdigest() == expected["whole_raw_sha256"], "Whole raw SHA mismatch")
    gz_hash = file_digest(payload)
    result = {
        "status":"PASS_REGENERATION", "total_records":total, "exact11_records":exact11,
        "raw_byte_count":total*8, "whole_raw_sha256":whole.hexdigest(),
        "gzip_byte_count":payload.stat().st_size, "gzip_file_sha256":gz_hash,
        "historical_gzip_byte_match":gz_hash == expected["gzip_file_sha256"],
        "metadata_sha256":file_digest(metadata_path), "generator_sha256":file_digest(Path(__file__)),
        "python":sys.version, "numpy":np.__version__, "zlib":zlib.ZLIB_RUNTIME_VERSION,
        "blocks":blocks,
    }
    with (output_dir / "regeneration.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; an existing path is refused")
    parser.add_argument("--batch-records", type=int, default=750_000)
    args = parser.parse_args()
    result = generate(args.output_dir.resolve(), args.batch_records)
    print(json.dumps({k:result[k] for k in ("status", "total_records", "whole_raw_sha256", "gzip_file_sha256", "historical_gzip_byte_match")}, indent=2))

if __name__ == "__main__":
    main()
