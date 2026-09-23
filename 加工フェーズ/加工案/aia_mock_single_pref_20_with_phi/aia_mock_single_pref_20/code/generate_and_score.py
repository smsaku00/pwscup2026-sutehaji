"""Rebuild single-prefecture team 28 AIA mocks and their Monte Carlo phi.

Inputs are the team's *private* main B_self and A_bg.  No B or B_self bytes are
included in the output package.  All IDs are synthetic and are not contest IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import csr_matrix

QCOL = ["age", "SBP", "TG", "HDL", "ALT"]
PUBLIC = ["challenge_row_id", *QCOL, "sex", "prefecture"]
GROUP = ["sex", "prefecture", "age_band", "onset"]
PANEL_COUNTS = {"main":10, "prelim":5, "practice":5}
N_TRIALS = 1000
PREFECTURE = "神奈川県"
SEED = 20260923


def zip_csv(z: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(filename)), float_precision="round_trip")


def hash_file(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def rare_proxy(df: pd.DataFrame, schema: dict) -> np.ndarray:
    detector = schema["rare_detector"]
    diff = df[detector["columns"]].to_numpy(float) - np.asarray(detector["mu"])
    md2 = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(detector["sigma"]), diff)
    return md2 >= detector["band"]["lo"] ** 2


def choose_members(b: pd.DataFrame, a: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Exact 163/16/13/4 quotas, capacity constraints for exact pair matching."""
    n = len(b)
    incidence = []
    capacities = []
    cap = a.groupby(GROUP).size().to_dict()
    for key, indices in b.groupby(GROUP).groups.items():
        row = np.zeros(n)
        row[np.asarray(indices, dtype=int)] = 1
        incidence.append(row)
        capacities.append(cap.get(key, 0))
    rare = b.is_rare.to_numpy(int)
    onset = b.onset.to_numpy(int)
    quotas = np.array([163, 16, 13, 4])
    matrix = csr_matrix(np.vstack([np.ones(n), onset, rare, rare * onset, *incidence]))
    lo = np.r_[quotas, np.zeros(len(capacities))]
    hi = np.r_[quotas, capacities]
    result = milp(rng.random(n), integrality=np.ones(n),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(matrix, lo, hi),
                  options={"time_limit": 90})
    if result.x is None or result.status != 0:
        raise RuntimeError(f"No feasible target selection: {result.message}")
    selected = np.flatnonzero(result.x > .5)
    assert len(selected) == 163
    return selected


def pair_controls(b: pd.DataFrame, a: pd.DataFrame, selected: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    grouped = a.groupby(GROUP).groups
    for key, indices in b.loc[selected].groupby(GROUP).groups.items():
        bi = np.asarray(indices, dtype=int)
        ai = np.asarray(grouped[key], dtype=int)
        mismatch = (b.loc[bi, "proxy"].to_numpy()[:, None] !=
                    a.loc[ai, "proxy"].to_numpy()[None, :])
        age_gap = np.abs(b.loc[bi, "age"].to_numpy()[:, None] -
                         a.loc[ai, "age"].to_numpy()[None, :])
        cost = 100 * mismatch + age_gap + rng.random((len(bi), len(ai))) * 1e-3
        rows, cols = linear_sum_assignment(cost)
        assert len(rows) == len(bi)
        pairs.extend((int(bi[i]), int(ai[j])) for i, j in zip(rows, cols))
    member_ids, control_ids = np.asarray(pairs).T
    assert len(np.unique(member_ids)) == len(np.unique(control_ids)) == 163
    return member_ids, control_ids


def reference_success(background: pd.DataFrame, query: pd.DataFrame,
                      rng: np.random.Generator) -> np.ndarray:
    """1-NN of 1000 uniform size-1049 samples; reference block and A_bg SD.

    Nearest continuous-QI ties are averaged as in the starter reference.
    Across a sample, an empty (sex, prefecture) block falls back to sex and
    then all candidates.  The full sorted candidate list is checked per draw.
    """
    assert background.index.equals(pd.RangeIndex(len(background)))
    sample_mask = np.zeros((N_TRIALS, len(background)), dtype=bool)
    for mask in sample_mask:
        mask[rng.choice(len(background), 1049, replace=False)] = True
    std = background[QCOL].std().to_numpy(float)
    std = np.where(std > 0, std, 1.0)
    ax = background[QCOL].to_numpy(float) / std
    qx = query[QCOL].to_numpy(float) / std
    times = background.time.to_numpy(float)
    true_times = query.time.to_numpy(float)
    blocks = background.groupby(["sex", "prefecture"]).indices
    sexes = background.groupby("sex").indices
    successes = np.zeros((N_TRIALS, len(query)), dtype=bool)
    for j, key in enumerate(query[["sex", "prefecture"]].itertuples(index=False, name=None)):
        idx = blocks.get(key)
        if idx is None or not len(idx):
            idx = sexes.get(key[0])
        if idx is None or not len(idx):
            idx = np.arange(len(background))
        dist = np.mean((ax[idx] - qx[j]) ** 2, axis=1)
        order = idx[np.argsort(dist, kind="stable")]
        available = sample_mask[:, order]
        has_one = available.any(axis=1)
        if not has_one.all():
            # Reference search for samples where the primary block is empty.
            sex_idx = sexes.get(key[0])
            if sex_idx is None or not len(sex_idx):
                sex_idx = np.arange(len(background))
            sex_dist = np.mean((ax[sex_idx] - qx[j]) ** 2, axis=1)
            sex_order = sex_idx[np.argsort(sex_dist, kind="stable")]
            order_missing = sex_order
        nearest = order[np.argmax(available, axis=1)]
        if not has_one.all():
            nearest[~has_one] = order_missing[
                np.argmax(sample_mask[~has_one][:, order_missing], axis=1)]
        # QI ties are improbable but use the reference's mean of tied time.
        predicted = times[nearest].copy()
        if np.unique(dist).size != len(dist):
            unique_nn, inv = np.unique(nearest, return_inverse=True)
            for k, candidate in enumerate(unique_nn):
                if candidate not in idx:
                    continue
                candidate_d = np.mean((ax[candidate] - qx[j]) ** 2)
                ties = idx[dist == candidate_d]
                if len(ties) > 1:
                    for row in np.flatnonzero(inv == k):
                        in_sample = ties[sample_mask[row, ties]]
                        if len(in_sample) > 1:
                            predicted[row] = np.mean(np.sort(times[in_sample]))
        successes[:, j] = np.abs(predicted - true_times[j]) <= .5
    return successes


def generate(main_zip: Path, prelim_zip: Path, practice_zip: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(main_zip) as z:
        b_all = zip_csv(z, "B_self.csv")
        main_background = zip_csv(z, "A_bg.csv")
        schema = json.loads(z.read("schema.json"))
    with zipfile.ZipFile(prelim_zip) as z:
        prelim_background = zip_csv(z,"A_bg.csv")
    with zipfile.ZipFile(practice_zip) as z:
        practice_background = zip_csv(z,"pwscup2026-kit-main/participant_data/A_bg.csv")
    sources = {"main":main_background,"prelim":prelim_background,
               "practice":practice_background}
    assert len(b_all) == 1049 and all(len(frame)==5143 for frame in sources.values())
    b_local = b_all[b_all.prefecture == PREFECTURE].copy().reset_index(names="b_index")
    assert len(b_local) == 387

    # P_i itself is secret. Pick a fixed, plausible 313-person local block;
    # keep all 14 known local rare patients so panels can omit any one of them.
    p_rng = np.random.default_rng(SEED)
    rare_rows = np.flatnonzero(b_local.is_rare.eq(1).to_numpy())
    normal_rows = np.flatnonzero(b_local.is_rare.eq(0).to_numpy())
    p_rows = np.r_[rare_rows, p_rng.choice(normal_rows, 313 - len(rare_rows), replace=False)]
    b = b_local.iloc[p_rows].reset_index(drop=True).copy()
    b["age_band"] = (b.age // 10).astype(int)
    b["proxy"] = rare_proxy(b, schema).astype(int)
    private_root = output / "private"
    private_root.mkdir(exist_ok=True)
    (output / "public").mkdir(exist_ok=True)
    # Keep this away from public/main's reduced A_bg files: their difference
    # reveals the main controls.  Users already have their original main A_bg.
    main_background.to_csv(private_root / "A_bg_main_original.csv",
                           index=False,float_format="%.17g")
    b[["b_index", "record_id", "sex", "age", "is_rare", "onset"]].to_csv(
        private_root / "proxy_P_313.csv", index=False)
    records = []
    for scenario,panel in ((s,p) for s,count in PANEL_COUNTS.items()
                           for p in range(1,count+1)):
        a_all = sources[scenario]
        a = a_all[a_all.prefecture == PREFECTURE].copy().reset_index(names="a_index")
        a["age_band"] = (a.age // 10).astype(int)
        a["proxy"] = rare_proxy(a, schema).astype(int)
        seed = SEED + {"main":0,"prelim":1000,"practice":2000}[scenario] + panel
        rng = np.random.default_rng(seed)
        selected = choose_members(b, a, rng)
        bi, ai = pair_controls(b, a, selected, rng)
        rows = pd.concat([b.loc[bi], a.loc[ai]], ignore_index=True)
        role = np.r_[np.repeat("member", 163), np.repeat("control", 163)]
        order = rng.permutation(326)
        query = rows.iloc[order].reset_index(drop=True)
        public = query[PUBLIC[1:]].copy()
        public.insert(0, "challenge_row_id", np.arange(326))
        truth = pd.DataFrame({
            "challenge_row_id": np.arange(326), "label": role[order],
            "time": query.time.to_numpy(), "onset": query.onset.to_numpy(),
            "member_true_rare_control_unknown": np.r_[b.loc[bi,"is_rare"].to_numpy(int),
                                                      np.full(163,-1)][order],
            "public_proxy": query.proxy.to_numpy(int),
        })
        mapping = pd.DataFrame({
            "challenge_row_id": np.arange(326),
            "source": np.r_[np.repeat("main_B",163), np.repeat(f"{scenario}_A_bg",163)][order],
            "source_row_index": np.r_[b.loc[bi,"b_index"].to_numpy(int),
                                      a.loc[ai,"a_index"].to_numpy(int)][order],
            "pair_id": np.tile(np.arange(163),2)[order],
        })
        # Main controls are withheld from main A_bg. Cross-world controls do
        # not occur in the unmodified main A_bg used by the submitted C.
        held_out = set(a.loc[ai, "a_index"].astype(int))
        background = (main_background.drop(index=sorted(held_out)).reset_index(drop=True)
                      if scenario=="main" else main_background.reset_index(drop=True))
        label_m = truth.label.eq("member").to_numpy()
        assert len(background) == (4980 if scenario=="main" else 5143) and len(held_out)==163
        assert public.shape == (326,8) and public.columns.to_list() == PUBLIC
        assert public.prefecture.eq(PREFECTURE).all()
        assert int(truth.loc[label_m,"onset"].sum()) == int(truth.loc[~label_m,"onset"].sum()) == 16
        assert int(truth.loc[label_m,"member_true_rare_control_unknown"].sum()) == 13
        assert int(truth.loc[label_m & truth.member_true_rare_control_unknown.eq(1),"onset"].sum()) == 4
        for member_idx, control_idx in zip(bi, ai):
            assert tuple(b.loc[member_idx, GROUP]) == tuple(a.loc[control_idx, GROUP])
        target_k = {"main":9400,"prelim":9500,"practice":9600}[scenario] + panel
        name = f"{scenario}/panel_{panel:02d}"
        pubdir = output / "public" / name
        privdir = private_root / name
        pubdir.mkdir(parents=True, exist_ok=True)
        privdir.mkdir(parents=True, exist_ok=True)
        challenge_path = pubdir / f"aia_challenge_{target_k}.csv"
        public.to_csv(challenge_path,index=False,float_format="%.17g")
        if scenario == "main":
            background.to_csv(pubdir / "A_bg.csv",index=False,float_format="%.17g")
        truth.to_csv(privdir / "truth.csv",index=False,float_format="%.17g")
        mapping.to_csv(privdir / "row_mapping.csv",index=False)

        successes = reference_success(background, query, np.random.default_rng(seed + 1_000_000))
        pm = successes[:,label_m].mean(axis=1)
        pc = successes[:,~label_m].mean(axis=1)
        r = np.clip((pm - pc) / np.maximum(1-pc,1e-9),0,1)
        run = pd.DataFrame({"trial":np.arange(1,N_TRIALS+1),"p_m":pm,"p_c":pc,"R":r})
        run.to_csv(privdir / "phi_1000_trials.csv",index=False,float_format="%.17g")
        entry = {"scenario":scenario,"panel":name,"mock_target_k":target_k,"prefecture":PREFECTURE,
                 "phi_mock":float(r.mean()),"phi_mc_se":float(r.std(ddof=1)/np.sqrt(N_TRIALS)),
                 "p_m_mean":float(pm.mean()),"p_c_mean":float(pc.mean()),
                 "members":163,"controls":163,"member_rare":13,"member_rare_onset":4,
                 "onset_each":16,"background_n":len(background),
                 "proxy_mismatched_pairs":int((b.loc[bi,"proxy"].to_numpy()!=a.loc[ai,"proxy"].to_numpy()).sum()),
                 "challenge_sha256":hash_file(challenge_path)}
        records.append(entry)
        print(f"{name} phi={entry['phi_mock']:.6f} SE={entry['phi_mc_se']:.6f}",flush=True)
    pd.DataFrame(records).to_csv(output / "phi_results.csv", index=False, float_format="%.17g")
    (output / "protocol.json").write_text(json.dumps({
        "team":28,"worlds":{"main":10,"prelim_controls":5,"practice_controls":5},
        "prefecture":PREFECTURE,
        "P_proxy": "fixed 313 of 387 same-prefecture B, 14 rare retained; hidden true P unknown",
        "target_selection": "independent random MILP objective subject to 163/16/13/4 and pair capacity",
        "controls": "main: held-out main A_bg; prelim/practice: separate-world controls",
        "control_rare_truth": "unknown in all three A_bg sources",
        "background": "main: 4980 after holdout; prelim/practice: original main 5143",
        "phi": "1000 independently drawn uniform 1049-row A_bg samples; starter 1-NN, QI standardized by panel A_bg SD, sex/pref block, tau=0.5, mean of clipped R",
        "phi_is_official":False,"seed":SEED,
        "input_sha256":{"main":hash_file(main_zip),"prelim":hash_file(prelim_zip),
                        "practice":hash_file(practice_zip)},
    },ensure_ascii=False,indent=2)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-zip",type=Path,required=True)
    parser.add_argument("--prelim-zip",type=Path,required=True)
    parser.add_argument("--practice-zip",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    generate(args.main_zip,args.prelim_zip,args.practice_zip,args.out)


if __name__ == "__main__":
    main()
