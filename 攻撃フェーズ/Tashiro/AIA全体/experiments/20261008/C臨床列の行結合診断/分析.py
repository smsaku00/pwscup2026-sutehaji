"""同性別内の臨床4列を列別に置換する後付け行結合診断。39pool外。

全44C×20変換の予測を保存した後でのみ公開予備truthを採点する。
原本C・固定既存方式・39候補は変更しない。B_self/本戦truthは読まない。
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[name]="1"
import argparse,hashlib,importlib.util,json,platform,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"属性値集合による選別"
CFG=json.loads((HERE/"固定条件.json").read_text())
spec=importlib.util.spec_from_file_location("fixed_column_values_1008",SOURCE/"分析.py")
col=importlib.util.module_from_spec(spec);spec.loader.exec_module(col)
core=col.core;stable=col.stable
COLS=CFG["clinical_columns"];METHODS=CFG["methods"];REPS=CFG["repeats"]
WORK=HERE/"作業用";OUT=HERE/"集計"
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","hit_m_rare","hit_c_rare"]
DERIVED=["signed_hit_gap","unclipped_R_all","zero_clip_uplift"]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def now():return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()

def prepare_background(q,bg,scale):
    db=col.distance(q,bg,scale);sorted_bg=np.sort(db,axis=1)
    return [(np.r_[0.,row[np.isfinite(row)]],np.arange(np.isfinite(row).sum()+1,dtype=float)) for row in sorted_bg],len(bg)

def baseline(q,c,scale,background):
    """既存continuous CDFと1e-10 full5同点解消。BG側だけ事前計算。"""
    dc=col.distance(q,c,scale);prepared,n=background;pair=np.empty_like(dc)
    for i,(xp,fp) in enumerate(prepared):
        interpolated=np.interp(dc[i],xp,fp,right=len(xp)-1)
        pair[i]=-np.log((interpolated+.25)/(n+1))
    pair[~np.isfinite(dc)]=-1e4
    full=cdist(core.raw(q)/scale,core.raw(c)/scale,"sqeuclidean")/5
    pair-=stable.EPSILON*np.minimum(full,1000.)
    selected=stable.select(pair,dc,q.sex.to_numpy(),c.sex.to_numpy(),"sex_quota")
    return selected,pair,dc

def shuffled(c,rng):
    transformed=c.copy(deep=True);mapping=hashlib.sha256()
    for sex in sorted(c.sex.unique()):
        ix=np.flatnonzero(c.sex.to_numpy()==sex)
        for name in COLS:
            order=rng.permutation(ix);mapping.update(order.astype('<i8').tobytes())
            transformed.loc[ix,name]=c.loc[order,name].to_numpy()
            assert np.array_equal(np.sort(c.loc[ix,name].to_numpy()),np.sort(transformed.loc[ix,name].to_numpy()))
    rest=[name for name in c if name not in COLS]
    assert transformed[rest].equals(c[rest])
    return transformed,mapping.hexdigest()

def predictions(q,c,bg,scale,background):
    base,pair,dc=baseline(q,c,scale,background);ev=col.evidence(q,c,bg)
    masks=np.stack([base]+[col.independent_select(q,ev,name) for name in METHODS[1:]])
    return masks,ev,pair,dc

def predict():
    start=time.time();WORK.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    assert not (HERE/"予測実行記録.json").exists(),"保存済み予測は上書きしない"
    assert METHODS==col.METHODS[:3] and COLS==col.COLS and REPS==20
    entries=[];invariance=[];preservation=[];repro=[];changes=[]
    source_hashes={str(p):sha(p) for p in [SOURCE/"分析.py",SOURCE/"固定条件.json"]+
        [col.OLD/name for name in ["分析.py","同点に依存しない対応.py","対応競合の頑健性.py","対応競合の分析.py"]]}
    ci=col.CONTEXTS.index("full_reference_overlap")
    for phase,folder,bgp in col.SETS:
        bg=core.read(bgp);raw=core.raw(bg);scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem[2:])):
            team=int(cp.stem[2:]);c=core.read(cp);q=core.read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            assert len(c)==1049 and len(q)==326
            prepared=prepare_background(q,bg,scale);original,ev,pair,dc=predictions(q,c,bg,scale,prepared)
            reference_pair,reference_dc=col.baseline_pair(q,c,bg,scale)
            assert np.array_equal(reference_dc,dc) and np.array_equal(reference_pair,pair)
            saved_cache=SOURCE/f"作業用/{phase}_{team}.npz";source_hashes[str(saved_cache)]=sha(saved_cache)
            with np.load(saved_cache) as previous:
                assert np.array_equal(original,previous["real"][ci,:3])
                assert np.array_equal(original[0],previous["old_full"])
            repro.append(dict(phase=phase,team=team,baseline_pair_exact=True,baseline_mask_changed=0,column_mask_changed=0))
            rng=np.random.default_rng(CFG["seed_base"]+1000*team+(phase=="main")*100000);repeats=[]
            for repeat in range(REPS):
                transformed,mapping=shuffled(c,rng);selected,new_ev,_,_=predictions(q,transformed,bg,scale,prepared)
                for key in ["dc","db","ratio","all4","top2","exact"]:assert np.array_equal(ev[key],new_ev[key]),key
                assert np.array_equal(original[1:],selected[1:])
                repeats.append(selected)
                invariance.append(dict(phase=phase,team=team,repeat=repeat,evidence_all6_arrays_exact=True,
                    all4_selected_changed=0,top2_selected_changed=0))
                preservation.append(dict(phase=phase,team=team,repeat=repeat,mapping_sha256=mapping,
                    same_sex_clinical_value_multisets_preserved=True,other_columns_unchanged=True,
                    clinical_cells_changed=int((c[COLS].to_numpy()!=transformed[COLS].to_numpy()).sum()),
                    rows_with_clinical_changes=int((c[COLS].to_numpy()!=transformed[COLS].to_numpy()).any(axis=1).sum())))
                for mi,method in enumerate(METHODS):
                    for sex in [-1,0,1]:
                        use=np.ones(len(q),bool) if sex==-1 else q.sex.to_numpy()==sex
                        a,b=original[mi,use],selected[mi,use];union=int((a|b).sum())
                        changes.append(dict(phase=phase,team=team,repeat=repeat,method=method,sex=sex,q_n=int(use.sum()),
                            selected_original=int(a.sum()),selected_shuffled=int(b.sum()),selected_changed=int((a!=b).sum()),
                            jaccard=float((a&b).sum()/union) if union else 1.))
            cache=WORK/f"{phase}_{team}.npz";np.savez_compressed(cache,original=original,shuffled=np.stack(repeats))
            entries.append(dict(phase=phase,team=team,path=str(cache.relative_to(HERE)),sha256=sha(cache)))
            print(json.dumps(dict(phase=phase,team=team,seconds=round(time.time()-start,1))),flush=True)
    assert len(entries)==44 and len(invariance)==44*REPS
    for path,expected in source_hashes.items():assert sha(path)==expected
    pd.DataFrame(invariance).to_csv(OUT/"値集合の証拠と選別の厳密不変.csv",index=False)
    pd.DataFrame(preservation).to_csv(OUT/"性別内周辺値と非臨床列の保存.csv",index=False)
    pd.DataFrame(repro).to_csv(OUT/"既存3方式の全44C再現.csv",index=False)
    ch=pd.DataFrame(changes);ch.to_csv(OUT/"全44C_選別変化とJaccard.csv",index=False)
    ch[ch.phase.eq("main")].to_csv(OUT/"本戦_選別変化とJaccardのみ.csv",index=False)
    dump(HERE/"予測実行記録.json",dict(predictions_completed_at=now(),seconds=time.time()-start,code_sha256=sha(__file__),
        protocol_sha256=sha(HERE/"固定条件.json"),input_sha256=dict(core.HASHES),source_dependency_sha256=source_hashes,
        predictions=entries,C_count=44,repeats=20,all_transformed_predictions_completed=True,
        public_prelim_truth_read=False,main_truth_read=False,B_self_read=False,original_C_modified=False,
        submission_created=False,initial_39_pool_changed=False,python=sys.executable,python_version=platform.python_version()))

def add_derived(d):
    d["signed_hit_gap"]=d.hit_m-d.hit_c
    d["unclipped_R_all"]=(d.p_m-d.p_c)/np.maximum(1-d.p_c,1e-9)
    d["zero_clip_uplift"]=d.r_all-d.unclipped_R_all
    return d

def score():
    rec=json.loads((HERE/"予測実行記録.json").read_text())
    assert rec["code_sha256"]==sha(__file__) and rec["protocol_sha256"]==sha(HERE/"固定条件.json")
    assert rec["C_count"]==44 and rec["all_transformed_predictions_completed"]
    for item in rec["predictions"]:assert sha(HERE/item["path"])==item["sha256"]
    td=next((col.ROOT/"予備戦/配布データ/結果").glob("正解*"));truth_hashes={};rows=[]
    for item in rec["predictions"]:
        if item["phase"]!="prelim":continue
        team=item["team"];path=td/f"aia_truth_{team}.csv";truth=core.read(path).sort_values("challenge_row_id").reset_index(drop=True);truth_hashes[str(path)]=sha(path)
        with np.load(HERE/item["path"]) as data:
            for mi,method in enumerate(METHODS):
                rows.append(dict(team=team,method=method,source="original_C",repeat=-1,**core.scores(np.where(data["original"][mi],9.5,0),truth)))
                for repeat in range(REPS):rows.append(dict(team=team,method=method,source="independent_clinical_column_shuffle",repeat=repeat,
                    **core.scores(np.where(data["shuffled"][repeat,mi],9.5,0),truth)))
    table=add_derived(pd.DataFrame(rows));table.to_csv(OUT/"原本Cと臨床列交換_予備対象別と20反復.csv",index=False)
    old=pd.read_csv(SOURCE/"集計/予備戦_対象別と20反復null.csv")
    old=old[old.context.eq("full_reference_overlap")&old.source.eq("real_C")&old.method.isin(METHODS)]
    current=table[table.source.eq("original_C")].set_index(["team","method"]).sort_index()
    old=old.set_index(["team","method"]).sort_index();assert np.max(np.abs(current[METRICS].to_numpy()-old[METRICS].to_numpy()))<1e-12
    diffs=[]
    for (team,method),g in table.groupby(["team","method"]):
        base=g[g.source.eq("original_C")].iloc[0];sh=g[g.source.ne("original_C")]
        row=dict(team=int(team),method=method,repeats=20,original_R=float(base.r_all),shuffled_R_mean=float(sh.r_all.mean()),
            shuffled_R_std=float(sh.r_all.std(ddof=0)),shuffled_R_min=float(sh.r_all.min()),shuffled_R_max=float(sh.r_all.max()))
        for field in METRICS+DERIVED:row["delta_"+field]=float(sh[field].mean()-base[field])
        diffs.append(row)
    dd=pd.DataFrame(diffs);dd.to_csv(OUT/"臨床列交換の影響_予備対象別.csv",index=False)
    mean=[];spread=[];repeats=[]
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        use=table[~table.team.isin(excluded)]
        m=use.groupby(["method","source"])[METRICS+DERIVED].mean().reset_index();m.insert(0,"group",group);mean.append(m)
        rr=use[use.source.ne("original_C")].groupby(["method","repeat"])[METRICS+DERIVED].mean().reset_index();rr.insert(0,"group",group);repeats.append(rr)
        for method,g in dd[~dd.team.isin(excluded)].groupby("method"):
            spread.append(dict(group=group,method=method,targets=len(g),original_R=float(g.original_R.mean()),shuffled_R=float(g.shuffled_R_mean.mean()),
                delta_R_mean=float(g.delta_r_all.mean()),delta_R_target_min=float(g.delta_r_all.min()),delta_R_target_q25=float(g.delta_r_all.quantile(.25)),
                delta_R_target_median=float(g.delta_r_all.median()),delta_R_target_q75=float(g.delta_r_all.quantile(.75)),delta_R_target_max=float(g.delta_r_all.max()),
                delta_signed_hit_gap_mean=float(g.delta_signed_hit_gap.mean()),delta_signed_hit_gap_total=float(g.delta_signed_hit_gap.sum()),
                delta_hit_m_total=float(g.delta_hit_m.sum()),delta_hit_c_total=float(g.delta_hit_c.sum()),
                delta_unclipped_R_mean=float(g.delta_unclipped_R_all.mean()),delta_zero_clip_uplift_mean=float(g.delta_zero_clip_uplift.mean()),
                improved_targets=int(g.delta_r_all.gt(1e-12).sum()),worsened_targets=int(g.delta_r_all.lt(-1e-12).sum()),
                equal_targets=int(g.delta_r_all.abs().le(1e-12).sum()),independent_validation=False))
    pd.concat(mean).to_csv(OUT/"予備戦_20対象と18対象の平均.csv",index=False)
    pd.concat(repeats).to_csv(OUT/"予備戦_反復ごとの全体平均.csv",index=False)
    pd.DataFrame(spread).to_csv(OUT/"臨床列交換の影響_平均と対象間ばらつき.csv",index=False)
    dump(HERE/"採点実行記録.json",dict(scored_at=now(),prediction_receipt_sha256=sha(HERE/"予測実行記録.json"),
        code_sha256=sha(__file__),protocol_sha256=sha(HERE/"固定条件.json"),public_prelim_truth_sha256=truth_hashes,
        all44x20_transformed_predictions_saved_before_truth_read=True,saved_original_63_scores_match=True,
        phi_unknown=True,main_truth_read=False,B_self_read=False,original_C_modified=False,initial_39_pool_changed=False,
        real_team_processing_not_inferred=True,independent_validation=False,positive_clipped_R_not_leakage_proof=True,
        output_sha256={p.name:sha(p) for p in OUT.glob("*.csv")}))
    print(pd.DataFrame(spread).to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["predict","score"],required=True)
    {"predict":predict,"score":score}[ap.parse_args().stage]()
