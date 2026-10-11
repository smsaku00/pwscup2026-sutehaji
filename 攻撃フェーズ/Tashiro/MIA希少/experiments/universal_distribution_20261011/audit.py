#!/usr/bin/env python3
"""Read-only reproduction of the abstract universal-distribution certificates.

Only candidate slots 0..10 and count capacities 0/1/2 are used.  No contest CSV,
official kit, candidate record ID, solver, or certificate-generation code is
imported.  All saved trees are re-evaluated on independently generated states.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
from itertools import combinations, product
import json
from math import comb, factorial
from pathlib import Path
import sys
from time import monotonic

if not sys.dont_write_bytecode:
    raise SystemExit("Use Python 3.12 with -B; no bytecode cache may be created")

import numpy as np

import adaptive_lift as lift
import adaptive_three as adaptive
import binary_bounds as bounds
import enumeration
import fixed_certificates as fixed
import scalar_bounds as scalar

HERE = Path(__file__).resolve().parent
MIA = HERE.parents[1]
DEFAULT_DATA = MIA / "analysis/universal_distribution_20261011"
QUESTIONS = ((3, 4, 5, 7, 8), (0, 3, 6, 8, 10), (0, 2, 5, 8, 9),
             (0, 1, 5, 6, 7), (0, 1, 2, 3, 4))
TYPE_ORDER = [(a, b) for b in range(6) for a in range(12-2*b)]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    return enumeration.hash_file(path)


def read(path, sources):
    path = path.resolve()
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def rate(n, m, h):
    require(n in (1, 2) and 5 <= m <= 18 and 0 <= h <= n,
            "Outside the documented binary rare-score domain")
    return Fraction((24-n)*h, 100*n*(m-h))


def normalized_type(row):
    a, b = row["a"], row["b"]
    ms = row["M_1person_targets_in_ID_order"]+row["M_2person_targets_in_ID_order"]
    return {"a": a, "b": b, "counts": [1]*a+[2]*b, "M": ms,
            "padding": [m-5 for m in ms], "response_count": row["independent_response_count"],
            "minimum_gap_exact": row["minimum_gap_exact"],
            "full_assignment_count": factorial(11)//(2**b*factorial(11-a-2*b))}


def verify_provenance(data, sources):
    provenance = read(data/"provenance.json", sources)
    count = 0
    for entry in provenance["original_files"]:
        path = (data/entry["path"]).resolve()
        require(path.is_relative_to(data), "Provenance path escapes the proof bundle")
        require(entry["transformation"] == "NONE_BYTE_IDENTICAL" and entry["sha256"] == entry["source_sha256"]
                and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "Preserved original certificate bytes/hash differ")
        sources[str(path)] = entry["sha256"]
        count += 1
    for item in provenance["embedded_historical_hash_maps"]:
        if item["original_field"] not in ("old_sources_sha256", "original_hashes"):
            continue
        for filename, expected in item["hashes"].items():
            require(Path(filename).name == filename, "Historical solver hash key is not a filename")
            path = MIA/"experiments/fast_solver_20261007"/filename
            require(sha(path) == expected, "Saved certificate names a different historical solver source")
            sources[str(path.resolve())] = expected
    return {"status":"PASS", "byte_identical_original_files":count}


def verify_fixed5(data, sources):
    report = read(data/"fixed5/results.json", sources)
    exact = read(data/"fixed5/exact_answers.json", sources)
    require(report["status"] == "PASS" and report["known_count_stage"] is True,
            "Fixed-five saved certificate did not pass or counts are not known")
    require(report["fixed_candidate_questions"] == [list(q) for q in QUESTIONS],
            "The adopted fixed-five candidate questions differ")
    require([(r["a"], r["b"]) for r in report["types"]] == TYPE_ORDER,
            "Fixed-five count-type coverage differs")
    require([(r["a"], r["b"]) for r in exact["types"]] == TYPE_ORDER,
            "Saved rational-answer type coverage differs")
    bits = [tuple(int(slot in q) for q in QUESTIONS) for slot in range(11)]
    signature_counts = {}
    for n in (1, 2):
        signatures = [tuple(sum(bits[i][q] for i in group) for q in range(5))
                      for group in combinations(range(11), n)]
        require(len(set(signatures)) == comb(11, n), "Adopted candidate signatures collide")
        signature_counts[str(n)] = len(signatures)
    rows = {}
    total = 0
    minimum_gap = None
    labelled = exact11 = offset = 0
    for entry, saved in zip(report["types"], exact["types"]):
        p = normalized_type(entry)
        a, b = p["a"], p["b"]
        rows[a, b] = p
        count = comb(22, b)*comb(22-b, a)
        require(entry["labelled_patterns"] == count and entry["zero_targets"] == 22-a-b
                and entry["assigned_candidates"] == a+2*b and entry["unassigned_candidates"] == 11-a-2*b,
                "Labelled pattern/cardinality definition differs")
        require(entry["first_pattern_id"] == offset+1 and entry["last_pattern_id"] == offset+count,
                "Fixed-five pattern-ID block range differs")
        require(len(p["M"]) == a+b and all(5 <= m <= 18 for m in p["M"]), "Invalid M setting")
        require(p["padding"] == entry["padding_1person_targets"]+entry["padding_2person_targets"]
                and all(0 <= v <= 13 for v in p["padding"]), "Unavailable fixed-five padding")
        require(saved["counts"] == p["counts"] and saved["M"] == p["M"]
                and saved["positive_target_count"] == a+b, "Rational-answer type parameters differ")
        alphabet = {}
        for hs in product(*(range(n+1) for n in p["counts"])):
            mean = (sum((rate(n, m, h) for n, m, h in zip(p["counts"], p["M"], hs)), Fraction())/(a+b)
                    if a+b else Fraction())
            require(hs not in alphabet, "Duplicate count vector")
            alphabet[hs] = mean
        require(len(alphabet) == len(set(alphabet.values())) == p["response_count"] == 2**a*3**b,
                "A complete exact one-round answer alphabet collides")
        saved_alphabet = {tuple(answer["h"]): Fraction(answer["mean"]) for answer in saved["answers"]}
        require(len(saved_alphabet) == len(saved["answers"]) and saved_alphabet == alphabet,
                "Saved exact answer values differ from the independently derived formula")
        if a+b:
            means = sorted(alphabet.values())
            gap = min(right-left for left, right in zip(means, means[1:]))
            require(str(gap) == p["minimum_gap_exact"] == saved["minimum_gap_exact"], "Minimum mean gap differs")
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            total += len(alphabet)
        else:
            require(entry["fixed_five_questions"] == "TRIVIAL_NO_MEMBERS" and p["minimum_gap_exact"] is None,
                    "Zero type is not trivial")
        labelled += count
        exact11 += count if a+2*b == 11 else 0
        offset += count
    require(total == 13_103 and labelled == 82_995_960 and exact11 == 49_402_850,
            "Independent answer/pattern totals differ")
    require(report["summary"]["independent_mean_answers_recomputed"] == total
            and report["summary"]["minimum_mean_gap_exact"] == str(minimum_gap), "Fixed-five summary differs")
    return rows, {"status": "PASS", "types": 42, "nonzero_types": 41,
                  "nonzero_exact_answers": total, "including_zero_exact_answers": total+1,
                  "local_signature_counts": signature_counts, "collisions": 0,
                  "minimum_gap_exact": str(minimum_gap), "labelled_patterns": labelled,
                  "exact11_patterns": exact11,
                  "proof": "Every round mean recovers the full target count vector; each target's five counts uniquely identify its one- or two-candidate membership. Permuting target labels together with their M values preserves the mean. The rank within one-member and two-member labels assigns exactly one M and common five-round padding to every labelled count vector."}


def verify_fixed4(data, proof_types, sources):
    search = read(data/"fixed4/search.json", sources)
    optimized = read(data/"fixed4/optimized_b4_m11.json", sources)
    saved_audit = read(data/"fixed4/independent_audit.json", sources)
    entries = search["types"]+[optimized]
    require(len(search["types"]) == 8, "Fixed-four bounded search has wrong scope")
    results = []
    for entry in entries:
        require(entry["status"] in ("SAT", "UNKNOWN"), "Bounded search status is invalid")
        if entry["status"] == "SAT":
            source = data/("fixed4/optimized_b4_m11.json" if entry is optimized else "fixed4/search.json")
            results.append(fixed.check_certificate(entry, proof_types[entry["a"], entry["b"]], source))
    require({(r["a"], r["b"]) for r in results} == {(6, 2), (7, 2), (5, 3), (3, 4)},
            "Four saved target-specific fixed-four SAT types differ")
    require(len(results) == 4 and sum(r["pair_state_count"] for r in results) == 461_340
            and sum(r["residual_keys_checked"] for r in results) == 473_220,
            "Fixed-four exhaustive pair/residual totals differ")
    require(saved_audit["summary"]["independently_verified_SAT_types"] == 4
            and saved_audit["summary"]["total_collisions"] == 0, "Fixed-four saved audit summary differs")
    return {"status": "PASS", "SAT_certificates": 4, "pair_states": 461_340,
            "residual_keys": 473_220, "collisions": 0, "certificates": results,
            "UNKNOWN_remains_UNKNOWN": True}


def verify_bounds(data, sources):
    saved = read(data/"fixed4/bounds.json", sources)
    recomputed = {"fixed4_binary_code_bound": bounds.exhaustive_subset_bound(),
                  "single_pair_adaptive_lower_bound": bounds.single_pair_adaptive_lower_bound(),
                  "balanced_all_single_fixed4": bounds.balanced_single_code(),
                  "type_classification_by_proved_bounds": bounds.type_bounds()}
    recomputed["complete_common_fixed4_classification"] = bounds.common_fixed4_classification(
        recomputed["balanced_all_single_fixed4"]["codewords_decimal"])
    for key, value in recomputed.items():
        require(value == saved[key], f"Independent fixed-four bound differs: {key}")
    anchor = read(data/"adaptive4/bounds/anchor_a1b1.json.gz", sources)
    cases = {row["candidate_count"]: row for row in anchor["scalar_boundary_cases"]}
    for n in (10, 11):
        lower = scalar.mathematical_pair_lower_bound(n)
        require(cases[n]["status"] == "UNSAT" and cases[n]["query_generator_is_complete"]
                and cases[n]["independent_mathematical_lower_bound"] == lower,
                "Scalar mathematical lower bound differs")
    nine = scalar.independently_check_pair_nine()
    for key in ("status", "state_count", "candidate_count", "depth", "cache_hits", "cache_misses", "nodes_by_remaining_depth"):
        # JSON converts histogram keys to strings.
        require(json.loads(json.dumps(nine[key])) == anchor["independent_pair9_exhaustive_check"][key],
                f"Independent scalar-nine exhaustive result differs: {key}")
    require(cases[9]["status"] == "UNSAT" and cases[9]["query_generator_is_complete"], "Scalar-nine saved solver is not exact UNSAT")
    eight = scalar.scalar_replay(cases[8]["tree"], scalar.scalar_states(8))
    require(cases[8]["status"] == "SAT" and eight["leaf_count"] == 28 and eight["maximum_queries"] <= 4,
            "Scalar-eight saved SAT tree fails independent replay")
    seven = scalar.fixed_seven_certificate()
    require(cases[7] == seven, "Fixed-seven boundary differs")
    unsat_types = [(a, b) for a, b in TYPE_ORDER if b and a+2*b <= 4]
    require(unsat_types == [(r["a"], r["b"]) for r in anchor["multi_target_fiber_lift"]["types"]],
            "Adaptive lower-bound lift type coverage differs")
    return {"status": "PASS", "fixed4_maximum_pair_separable_words": 7,
            "all_binary_word_subsets_tested": 65_536, "all_eleven_word_sets_tested": 4_368,
            "common_fixed4": recomputed["complete_common_fixed4_classification"]["summary"],
            "target_specific_fixed4_UNSAT_types_by_embedding": 20,
            "scalar_adaptive4_UNSAT_candidate_counts": [9, 10, 11],
            "scalar_eight_SAT_replay": eight, "scalar_seven_adaptive_search_skipped": True,
            "adaptive4_UNSAT_types": [list(t) for t in unsat_types],
            "scope": "Binary high/low subset/count questions after target counts are known; no impossibility claim for arbitrary multilevel confidence scores. Scalar-eight SAT is a conditional fiber result and is not lifted to an unknown global membership assignment."}


def verify_lift(data, sources):
    old_sources = {}
    manifest = lift.read(lift.OLD/"manifest.json", old_sources)
    historical = read(data/"adaptive4/a3b3/replay.json", sources)
    saved = read(data/"adaptive4/a4b3/replay.json", sources)
    require(manifest["caps"] == list(lift.CAPS_OLD) and manifest["q3_query"] == list(lift.Q3), "Historical abstract model differs")
    require(historical["overall_status"] == "SAT" and historical["total_states"] == 2_494_800, "Historical replay is incomplete")
    hashes = {row["q3_code"]: row["state_hash"] for row in historical["branches"]}
    unit_rates, mean_units, answers, denominator, gap = lift.exact_encoder()
    require(len(set(lift.WORDS)) == 11 and all(q.bit_count() == 5 for q in lift.EXTRA_QUERY), "Lift singleton code differs")
    seen, reports = set(), []
    nodes = 0
    for orbit in manifest["orbits"]:
        rep = orbit["representative"]
        certificate = lift.read(lift.OLD/"orbits"/str(rep)/"certificate.json.gz", old_sources)
        require(certificate["q3_code"] == rep and certificate["depth"] == 3, "Historical certificate header differs")
        source_rows = lift.generate_projection(rep)
        require(len(source_rows) == orbit["states"] and lift.state_hash(source_rows) == certificate["state_hash"],
                "Independently generated projection states differ")
        source_queries, source_codes, depths, count = lift.replay_representative(source_rows, certificate["tree"])
        nodes += count
        for code in orbit["q3_codes"]:
            require(code not in seen, "Duplicate historical initial branch")
            seen.add(code)
            mapping = orbit["mappings"][str(code)]
            pm, tm = mapping["candidate_old_to_new"], mapping["team_old_to_new"]
            require(sorted(pm) == list(range(11)) and sorted(tm) == list(range(6))
                    and all(lift.CAPS_OLD[t] == lift.CAPS_OLD[tm[t]] for t in range(6)), "Invalid historical transport")
            require(all((lift.Q3[t] >> i & 1) == (lift.Q3[tm[t]] >> pm[i] & 1) for t in range(6) for i in range(11)),
                    "Historical transport fails to preserve initial incidence")
            lookup = np.array([sum(1 << pm[i] for i in range(11) if mask >> i & 1) for mask in range(2048)], dtype=np.uint16)
            rows, queries, codes = np.zeros_like(source_rows), np.zeros_like(source_queries), np.zeros_like(source_codes)
            for old, new in enumerate(tm):
                rows[:, new] = lookup[source_rows[:, old]]
                queries[:, :, new] = lookup[source_queries[:, :, old]]
                codes += ((source_codes//np.uint16(3**old)) % np.uint16(3))*np.uint16(3**new)
            require(np.all(codes[:, 0] == code) and lift.state_hash(rows) == hashes[code], "Transported complete branch differs")
            result = lift.concrete_branch(rows, queries, codes, depths, code, unit_rates, mean_units, answers)
            result["representative"] = rep
            reports.append(result)
    total = sum(r["concrete_full_states"] for r in reports)
    require(total == 4_989_600 and len(reports) == 209 and seen == set(hashes), "Full lifted state coverage differs")
    saved_branches = {r["q3_code"]: r for r in saved["branches"]}
    for row in reports:
        require(row == saved_branches[row["q3_code"]], "Saved a4b3 concrete mean replay trace differs")
    sources.update({str((MIA/p).resolve()): digest for p, digest in old_sources.items()})
    for path, expected in saved["historical_sources_sha256"].items():
        marker = "/analysis/full_depth4_20261007/"
        if marker in path:
            current = lift.OLD/path.split(marker, 1)[1]
            require(sha(current) == expected, "Old lift certificate original source hash differs")
    return {"status": "PASS", "type": [4, 3], "total_states": total,
            "mean_responses_replayed": total*4, "collisions": 0, "maximum_queries": 4,
            "representative_certificates": len(manifest["orbits"]), "root_branches": len(reports),
            "representative_query_nodes": nodes, "minimum_gap_exact": str(gap),
            "exact_unit_denominator": denominator, "branches": reports,
            "candidate_record_ids_used": False}


def verify_adaptive(data, proof_types, sources):
    results = []
    saved_means = read(data/"adaptive4/mean_replay.json", sources)
    saved_types = {tuple(row["type"]): row for row in saved_means["types"]}
    for name in adaptive.TYPES:
        local_sources = {}
        run = data/"adaptive4"/name
        check, manifest = adaptive.readiness(name, run, local_sources)
        require(check["status"] == "READY_COMPLETE_SAT_CERTIFICATES", f"Required adaptive type {name} is incomplete")
        a, b, _ = adaptive.TYPES[name]
        proof = proof_types[a, b]
        portable_proof = {"M_1person_targets_in_ID_order": proof["M"][:a],
                          "M_2person_targets_in_ID_order": proof["M"][a:],
                          "independent_response_count": proof["response_count"], "minimum_gap_exact": proof["minimum_gap_exact"]}
        result = adaptive.verify_type(name, run, manifest, portable_proof, local_sources)
        require(result["branches"] == saved_types[a,b]["branches"],
                "Saved three-type concrete mean replay traces differ")
        sources.update({str((MIA/p).resolve()): digest for p, digest in local_sources.items()})
        results.append(result)
        print(json.dumps({"stage": "adaptive4", "type": [a,b], "states": result["total_states"], "status": "PASS"}), flush=True)
    lifted = verify_lift(data, sources)
    total = sum(r["total_states"] for r in results)+lifted["total_states"]
    require(total == 9_979_200, "Four adaptive type full-state total differs")
    saved_summary = read(data/"adaptive4/results.json", sources)
    require(saved_summary["summary"]["total_states"] == total,
            "Saved four-type adaptive summary differs")
    return {"status": "PASS", "types": 4, "total_states": total,
            "mean_responses_replayed": total*4, "collisions": 0,
            "maximum_queries": 4, "new_three_types": results, "a4b3_lift": lifted,
            "search_run": False, "fixed_SAT_types_searched": []}


def verify_classification(data, proof_types, sources):
    saved = read(data/"classification/results.json", sources)
    require([(r["a"], r["b"]) for r in saved["types"]] == TYPE_ORDER, "Classification coverage differs")
    fixed_sat = {(a,b) for a,b in TYPE_ORDER if b == 0 or b == 1 and a >= 8} | {(6,2),(7,2),(5,3),(3,4)}
    adaptive_unsat = {(a,b) for a,b in TYPE_ORDER if b and a+2*b <= 4}
    adaptive_sat = {(3,3),(4,3),(2,4),(0,5),(1,5)}
    counters = {"SAT": 0, "UNSAT": 0, "UNKNOWN": 0}
    for row in saved["types"]:
        a,b = key = row["a"],row["b"]
        fstatus = "SAT" if key in fixed_sat else "UNSAT" if b and a+2*b <= 9 else "UNKNOWN"
        require(row["target_specific_fixed4"] == fstatus, "Fixed-four classification lacks checked evidence")
        counters[fstatus] += 1
        astat = ("SKIPPED_FIXED_SAT" if key in fixed_sat else "UNSAT" if key in adaptive_unsat else
                 "SAT_OLD_CERTIFICATE" if key == (3,3) else "SAT_FULL_MEAN_REPLAY" if key in adaptive_sat else "UNRESOLVED")
        require(row["adaptive4"] == astat, "Adaptive-four classification lacks checked evidence")
        require(row["mean_response_count"] == proof_types[key]["response_count"], "Classification alphabet count differs")
    require(counters == {"SAT":18,"UNSAT":20,"UNKNOWN":4} and saved["summary"]["target_specific_fixed4"] == counters,
            "Classification summary differs")
    require(saved["summary"]["complete_adaptive_classification"] is False
            and saved["summary"]["minimum_queries_for_universal_guarantee_in_this_binary_model"] == 5,
            "Universal query conclusion differs")
    return {"status":"PASS", "fixed4":counters, "adaptive4_fixed_SAT_skipped":18,
            "adaptive4_UNSAT":4, "adaptive4_SAT":5, "adaptive4_UNRESOLVED":15,
            "universal_guarantee_queries_in_binary_model":5, "complete_adaptive_classification":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--enumeration-file", type=Path, help="Also audit every record of the separately released uint64.gz")
    parser.add_argument("--output-dir", type=Path, help="Create a new directory containing audit.json; existing directory is rejected")
    parser.add_argument("--chunk-records", type=int, default=262_144)
    args = parser.parse_args()
    require(sys.version_info[:2] == (3,12) and sys.dont_write_bytecode, "Use Python 3.12 with -B")
    require(1 <= args.chunk_records <= 2**20, "Chunk size must be bounded")
    if args.output_dir is not None:
        require(not args.output_dir.exists(), "Refusing to reuse or overwrite an existing output directory")
    data = args.data_dir.resolve()
    require(data.is_relative_to(MIA), "The proof bundle must be inside this MIA directory")
    sources = {}
    start = monotonic()
    provenance = read(HERE/"provenance.json", sources)
    packaging = verify_provenance(data, sources)
    for filename, expected in provenance["old_solver_original_hashes"].items():
        path = MIA/"experiments/fast_solver_20261007"/filename
        require(sha(path) == expected, "Historical Solver source hash differs")
        sources[str(path.resolve())] = expected
    proof_types, fixed5_report = verify_fixed5(data, sources)
    print(json.dumps({"stage":"fixed5", "status":"PASS", "exact_answers":13_103}), flush=True)
    fixed4_report = verify_fixed4(data, proof_types, sources)
    bounds_report = verify_bounds(data, sources)
    print(json.dumps({"stage":"fixed4_and_bounds", "status":"PASS"}), flush=True)
    adaptive_report = verify_adaptive(data, proof_types, sources)
    classification = verify_classification(data, proof_types, sources)
    report = {"status":"PASS", "model":"11 abstract candidates; 22 target labels; disjoint memberships of sizes 0/1/2; total <=11; counts known; binary subset questions; exact unrounded mean",
              "uses_competition_CSV_or_official_kit":False, "solver_imported":False,
              "packaging":packaging,
              "fixed5":fixed5_report, "fixed4":fixed4_report, "bounds":bounds_report,
              "adaptive4":adaptive_report, "classification":classification}
    if args.enumeration_file is not None:
        report["enumeration"] = enumeration.audit_saved_vectors(data/"fixed5/enumeration_manifest.json",
            args.enumeration_file.resolve(), [proof_types[k] for k in TYPE_ORDER], args.chunk_records)
    else:
        report["enumeration"] = {"status":"NOT_REQUESTED", "reason":"125 MiB release attachment is optional; count coverage and composition are checked by the basic audit"}
    for path, expected in sources.items():
        require(sha(Path(path)) == expected, "An input changed during the audit")
    report["inputs_sha256"] = {str(Path(path).relative_to(MIA)): digest for path,digest in sorted(sources.items())}
    report["source_sha256"] = {p.name:sha(p) for p in sorted(HERE.glob("*.py"))}
    report["python"] = sys.version
    report["numpy"] = np.__version__
    report["elapsed_seconds"] = monotonic()-start
    report["writes_files"] = args.output_dir is not None
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir/"audit.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    print(json.dumps({"status":"PASS", "fixed5_answers":13_103, "fixed4_SAT_certificates":4,
                      "adaptive4_states":9_979_200, "enumeration":report["enumeration"]["status"],
                      "writes_files":report["writes_files"], "elapsed_seconds":report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
