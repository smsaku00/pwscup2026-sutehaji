"""Public-gap-distribution sensitivity follow-up to P08D.

After observing the label-free full-background top10 collision gap median
~0.0057 and p90 ~0.0155, compare fixed smaller margin (.001,.005,.01)
with fixed penalties (.02,.05) at K10,20,40. This is a FOLLOW-UP planned
after the first P08D 18-method summary (all preliminary TPR unchanged);
therefore not independent validation even though the gap rationale uses
public P01 scores and all predictions precede truth reading.
Original 4-view 20/18 inference and official TPR evaluation code is REUSED
unchanged by dynamically supplying method-panel constants.
"""
from __future__ import annotations

import argparse,hashlib,importlib.util,json
from pathlib import Path

ORIGINAL="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_P08D_上位競合だけの疎な排他補正/run_p08d.py"
MARGINS=(.001,.005,.01)
PENALTIES=(.02,.05)
TOP_K=(10,20,40)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--root",required=True,type=Path)
    pa.add_argument("--out",required=True,type=Path)
    a=pa.parse_args()
    root=a.root.resolve()
    source=root/ORIGINAL
    spec=importlib.util.spec_from_file_location("p08d_pinned_lowgap",source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TOP_K==TOP_K
    assert module.MARGINS==(.02,.05,.10)
    assert module.PENALTIES==(.05,.15)
    module.MARGINS=MARGINS
    module.PENALTIES=PENALTIES
    module.METHODS=("P01",)+tuple(
      f"sparse_top{k}_margin{margin:g}_penalty{penalty:g}"
      for k in TOP_K for margin in MARGINS for penalty in PENALTIES
    )
    assert len(module.METHODS)==19
    # Original implementation is shared without edits and keeps the strict
    # prediction-score-freeze before truth-read sequencing.
    module.run(root,a.out.resolve())
    protocol=a.out.resolve()/"protocol.json"
    obj=json.loads(protocol.read_text())
    assert obj["official_prelim_truth_only_after_freeze"]
    obj.update({
       "study":"P08D small-gap public-score-distribution follow-up",
       "selection_basis":"top10 collision gap median~.0057, p90~.0155 measured label-free in earlier frozen scores",
       "post_first_score_exploration":True,
       "first_panel_all_TPR010_unchanged":True,
       "modified_constants":{"margin":MARGINS,"penalty":PENALTIES,"topk":TOP_K},
       "original_implementation_sha256":sha(source),
       "small_gap_wrapper_sha256":sha(Path(__file__).resolve()),
       "no_independent_generalization":True,
    })
    protocol.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
