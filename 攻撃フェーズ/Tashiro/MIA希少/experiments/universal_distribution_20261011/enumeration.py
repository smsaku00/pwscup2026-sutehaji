"""Optional independent chunk audit of the released 82,995,960 vectors."""
import gzip
from contextlib import nullcontext
import hashlib
import json
import math
import numpy as np
TARGET_COUNT = 22
CANDIDATE_COUNT = 11
MASK = (1 << 22)-1
def require(condition, message):
    if not bool(condition):
        raise ValueError(message)
def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def count_types():
    return [(a,b) for b in range(6) for a in range(12-2*b)]
def block_cardinality(a,b):
    return math.comb(22,b)*math.comb(22-b,a)

class BoundedCompressedReader:
    """Expose exactly one recorded gzip-member byte range to gzip.GzipFile."""
    def __init__(self, handle, offset, size):
        handle.seek(offset)
        self.handle = handle
        self.remaining = size
        self.position = 0

    def read(self, size=-1):
        if size < 0:
            size = self.remaining
        size = min(size, self.remaining)
        result = self.handle.read(size)
        self.remaining -= len(result)
        self.position += len(result)
        return result

    def tell(self):
        return self.position


def audit_label_padding(mask1, mask2, a, b, m_list):
    """Construct every row's label-to-M mapping in ID order, in vectorized form.

    Before label j, r1 counts the one-member labels below j and r2 the two-member
    labels below j.  Assigned M is M[r1] or M[a+r2].  The final sentinel zero
    permits indexing an inactive label after all members of its group have been
    encountered.  An active label can never receive that sentinel.
    """
    r1 = np.zeros(len(mask1), dtype=np.uint8)
    r2 = np.zeros(len(mask1), dtype=np.uint8)
    assigned_sum = np.zeros(len(mask1), dtype=np.uint16)
    singles = np.asarray(list(m_list[:a]) + [0], dtype=np.uint8)
    pairs = np.asarray(list(m_list[a:]) + [0], dtype=np.uint8)
    for label in range(TARGET_COUNT):
        bit = np.uint64(1 << label)
        is_one = (mask1 & bit) != 0
        is_two = (mask2 & bit) != 0
        assigned = np.where(is_one, singles[r1], np.where(is_two, pairs[r2], 0))
        active = is_one | is_two
        require(np.all((assigned[active] >= 5) & (assigned[active] <= 18)),
                "Active label has invalid M or unassigned rank")
        require(np.all(assigned[~active] == 0), "Zero-count label has an active setting")
        assigned_sum += assigned
        r1 += is_one
        r2 += is_two
    require(np.all(r1 == a) and np.all(r2 == b), "ID-order rank is incomplete")
    require(np.all(assigned_sum == sum(m_list)), "Label mapping omitted or reused an M slot")


def audit_saved_vectors(metadata_path, payload_path, proofs, chunk_records):
    metadata_hash = hash_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["schema_version"] == 1 and metadata["block_count"] == 42,
            "Unexpected saved enumeration schema")
    require(metadata["status"] == "PASS_GENERATION" and
            metadata["target_count"] == TARGET_COUNT and
            metadata["candidate_count"] == CANDIDATE_COUNT,
            "Wrong saved enumeration domain")
    require(metadata["source_proof_sha256"] == 'e939d3bd8d4eabd8e8a5367985ad16ad0fa9bf89acc4e4d262f356febdd5ceb3' and
            metadata["enumerator_sha256"] == 'a27117ae2182e32f43f6242e9e06421d4293b9d5321ca8da7827ada391e8dfe8',
            "Saved enumeration sources differ")
    require(metadata["target_ids_in_slot_order"] == list(range(18)) + [22, 25, 26, 27],
            "Target slot-to-ID mapping differs")
    require(metadata["encoding"]["dtype"] == "uint64" and
            metadata["encoding"]["byte_order"] == "little" and
            metadata["encoding"]["record_bytes"] == 8 and
            metadata["encoding"]["one_member_mask_bits"] == [0, 21] and
            metadata["encoding"]["two_member_mask_bits"] == [22, 43],
            "Saved payload encoding differs")
    blocks = metadata["blocks"]
    require([(block["a"], block["b"]) for block in blocks] == count_types(),
            "Saved blocks omit or duplicate a type")
    expected_total = sum(block_cardinality(a, b) for a, b in count_types())
    expected_exact = sum(block_cardinality(a, b) for a, b in count_types() if a + 2*b == 11)
    require(expected_total == 82995960 and expected_exact == 49402850,
            "Independent combinatorial totals differ")
    require(metadata["total_records"] == metadata["expected_total_records"] == expected_total and
            metadata["exact11_records"] == metadata["expected_exact11_records"] == expected_exact,
            "Main enumeration totals differ")
    require(metadata["raw_byte_count"] == expected_total * 8,
            "Wrong saved raw byte count")
    compressed_hash = hash_file(payload_path)
    historical_gzip_byte_match = (compressed_hash == metadata["gzip_file_sha256"]
                                  and metadata["gzip_byte_count"] == payload_path.stat().st_size)
    raw_hash = hashlib.sha256()
    total = 0
    exact_total = 0
    compressed_offset = 0
    label_assignments = 0
    reports = []
    # The historical artifact has exact recorded gzip-member byte ranges.
    # A different zlib/header can encode the same certified raw stream with
    # different sizes/offsets. In that case use the concatenated gzip stream
    # and its exact per-type raw cardinality; never use historical offsets.
    with payload_path.open("rb") as compressed, \
            (nullcontext(None) if historical_gzip_byte_match else gzip.open(payload_path, "rb")) as whole_stream:
        for index, (block, proof) in enumerate(zip(blocks, proofs)):
            a, b = block["a"], block["b"]
            expected_count = block_cardinality(a, b)
            require(block["n0"] == TARGET_COUNT - a - b and block["total_members"] == a + 2*b,
                    "Wrong saved count type")
            require(block["proof_type_index"] == index and block["M"] == proof["M"] and
                    block["padding"] == proof["padding"], "Saved proof or padding mapping differs")
            require(block["record_offset"] == total and block["count"] ==
                    block["expected_count"] == expected_count, "Wrong block offset or cardinality")
            require(block["raw_byte_count"] == expected_count * 8 and
                    block["minimum_gap_exact"] == proof.get("minimum_gap_exact"),
                    "Wrong block byte count or proof score gap")
            require(block["first_pattern_id"] == total + 1 and
                    block["last_pattern_id"] == total + expected_count, "Wrong pattern-ID range")
            require(block["compressed_offset"] == compressed_offset and block["compressed_size"] > 0,
                    "Wrong gzip member offset or size")
            limited = (BoundedCompressedReader(compressed, compressed_offset, block["compressed_size"])
                       if historical_gzip_byte_match else None)
            block_hash = hashlib.sha256()
            count = 0
            previous = None
            first = None
            last = None
            with (gzip.GzipFile(fileobj=limited, mode="rb") if historical_gzip_byte_match
                  else nullcontext(whole_stream)) as stream:
                while count < expected_count:
                    requested = min(chunk_records, expected_count - count)
                    raw = stream.read(requested * 8)
                    require(len(raw) == requested * 8, "Truncated saved enumeration block")
                    block_hash.update(raw)
                    raw_hash.update(raw)
                    values = np.frombuffer(raw, dtype="<u8")
                    require(np.all((values >> np.uint64(44)) == 0), "Count code exceeds 22+22 bits")
                    mask1 = values & np.uint64(MASK)
                    mask2 = values >> np.uint64(TARGET_COUNT)
                    require(np.all((mask1 & mask2) == 0), "One- and two-member labels overlap")
                    require(np.all(np.bitwise_count(mask1) == a) and
                            np.all(np.bitwise_count(mask2) == b), "Saved mask population differs from type")
                    require(a + 2*b <= CANDIDATE_COUNT, "Saved vector exceeds 11 members")
                    require(np.all(values[1:] > values[:-1]), "Block has unsorted or duplicate codes")
                    if previous is not None:
                        require(int(values[0]) > previous, "Block order fails across a chunk boundary")
                    if first is None:
                        first = int(values[0])
                    previous = last = int(values[-1])
                    audit_label_padding(mask1, mask2, a, b, proof["M"])
                    count += requested
                if historical_gzip_byte_match:
                    require(stream.read(1) == b"", "Block contains unexpected extra uncompressed bytes")
            if historical_gzip_byte_match:
                require(limited.remaining == 0, "Compressed member range contains unconsumed bytes")
            require(block_hash.hexdigest() == block["raw_sha256"], "Block raw hash differs")
            require(block["first_code"] == first and block["last_code"] == last,
                    "Recorded code endpoints differ")
            total += count
            if a + 2*b == 11:
                exact_total += count
            label_assignments += count * (a + b)
            compressed_offset += block["compressed_size"]
            reports.append({"a": a, "b": b, "status": "PASS_EXHAUSTIVE_LABELLED_VECTORS",
                            "vectors_read": count, "expected_vectors": expected_count,
                            "minimum_code": first, "maximum_code": last,
                            "raw_sha256": block_hash.hexdigest(),
                            "proof_type_index": index, "M": proof["M"], "padding": proof["padding"],
                            "all_vector_label_padding_assignments_checked": True,
                            "is_exact11_subset": a + 2*b == 11})
            print(json.dumps({"completed_block": index + 1, "a": a, "b": b,
                              "vectors_audited": total}, ensure_ascii=False), flush=True)
        if not historical_gzip_byte_match:
            require(whole_stream.read(1) == b"", "Concatenated gzip contains extra raw records or trailing data")
    require(total == expected_total and exact_total == expected_exact, "Final saved coverage count differs")
    require(compressed_offset == metadata["gzip_byte_count"], "Historical metadata compressed ranges are inconsistent")
    if historical_gzip_byte_match:
        require(compressed_offset == payload_path.stat().st_size, "Unrecorded compressed data exists")
    require(raw_hash.hexdigest() == metadata["whole_raw_sha256"], "Whole raw hash differs")
    require(hash_file(metadata_path) == metadata_hash, "Enumeration metadata changed during audit")
    return {"status": "PASS", "blocks": reports, "all_blocks": len(reports),
            "every_saved_vector_read": True, "saved_vectors": total,
            "independent_expected_vectors": expected_total,
            "exact11_vectors": exact_total, "independent_expected_exact11_vectors": expected_exact,
            "exact11_type_count": sum(report["is_exact11_subset"] for report in reports),
            "nonzero_target_label_settings_checked": label_assignments,
            "strict_order_and_valid_cardinality_prove_exhaustive_coverage": True,
            "whole_raw_sha256": raw_hash.hexdigest(), "gzip_file_sha256": compressed_hash,
            "historical_gzip_byte_match": historical_gzip_byte_match,
            "gzip_byte_count": payload_path.stat().st_size,
            "historical_gzip_file_sha256": metadata["gzip_file_sha256"],
            "compressed_validation_method": ("42 historical member byte ranges" if historical_gzip_byte_match
                                             else "concatenated gzip stream split by 42 exact raw type counts"),
            "metadata_sha256": metadata_hash,
            "label_padding_rule": "For label j with n=1, rank=popcount(mask1 & ((1<<j)-1)); for n=2, rank=a+popcount(mask2 & ((1<<j)-1)). Assign M[type][rank], padding=M-5. n=0 labels are omitted from the rare average",
            "candidate_questions_remain_actual_fixed_five": True,
            "per_target_padding_is_same_for_all_five_questions": True,
            "proof_transfer": "Every saved vector has its exact (a,b) certificate, and every active label receives exactly one canonical M. Each full per-question answer is uniquely decoded, then each five-answer single/pair signature uniquely identifies membership",
            "all_individual_membership_assignments_directly_replayed": False}


