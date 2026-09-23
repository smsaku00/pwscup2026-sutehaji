"""Check the distributed single-prefecture AIA panels and saved phi traces."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()
    root = args.dataset
    all_results = pd.read_csv(root / "phi_results.csv")
    assert len(all_results) == 20
    main_bg = pd.read_csv(root / "private/A_bg_main_original.csv")
    assert len(main_bg) == 5143
    qi = ["age","SBP","TG","HDL","ALT","sex","prefecture"]
    pair_keys = ["sex","prefecture","onset"]
    for row in all_results.itertuples():
        pubdir = root / "public" / row.panel
        privdir = root / "private" / row.panel
        chal = pd.read_csv(pubdir / f"aia_challenge_{row.mock_target_k}.csv")
        truth = pd.read_csv(privdir / "truth.csv")
        mapping = pd.read_csv(privdir / "row_mapping.csv")
        trace = pd.read_csv(privdir / "phi_1000_trials.csv")
        assert chal.shape == (326,8) and chal.prefecture.eq("神奈川県").all()
        assert chal.columns.to_list() == ["challenge_row_id",*qi]
        assert set(chal.columns).isdisjoint({"label","time","onset","is_rare"})
        assert chal.challenge_row_id.is_unique and len(trace) == 1000
        assert np.isclose(trace.R.mean(),row.phi_mock,atol=1e-14)
        assert np.allclose(trace.R, np.clip((trace.p_m-trace.p_c)/np.maximum(1-trace.p_c,1e-9),0,1))
        joined = chal.merge(truth,on="challenge_row_id",validate="one_to_one").merge(
            mapping,on="challenge_row_id",validate="one_to_one")
        target = joined.label.eq("member")
        assert target.sum() == (~target).sum() == 163
        assert joined.onset[target].sum() == joined.onset[~target].sum() == 16
        assert joined.loc[target,"member_true_rare_control_unknown"].sum() == 13
        assert joined.loc[target & joined.member_true_rare_control_unknown.eq(1),"onset"].sum() == 4
        assert joined.loc[~target,"member_true_rare_control_unknown"].eq(-1).all()
        a = joined.loc[target].set_index("pair_id")
        b = joined.loc[~target].set_index("pair_id")
        assert a.index.is_unique and b.index.is_unique and set(a.index) == set(b.index)
        for key in pair_keys:
            assert (a[key].sort_index().to_numpy()==b[key].sort_index().to_numpy()).all()
        assert (a.age.sort_index().to_numpy()//10 == b.age.sort_index().to_numpy()//10).all()
        control_q = joined.loc[~target,qi]
        if row.scenario == "main":
            panel_bg = pd.read_csv(pubdir / "A_bg.csv")
            assert len(panel_bg) == 4980
            reference = panel_bg
        else:
            assert not (pubdir/"A_bg.csv").exists()
            reference = main_bg
        assert len(control_q.merge(reference[qi],on=qi)) == 0
        print(f"{row.panel}: 326 rows; phi {row.phi_mock:.6f}; {row.scenario} controls absent from attacker background")
    print("All 20 panels passed.")


if __name__ == "__main__":
    main()
