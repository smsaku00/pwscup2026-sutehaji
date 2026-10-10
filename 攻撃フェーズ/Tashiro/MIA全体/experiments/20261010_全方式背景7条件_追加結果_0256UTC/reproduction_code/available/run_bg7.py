"""Frozen 7-background extension of exact P09/TAPAS implementations.

Original hyperparameters and Q holdout are unchanged. Prelim legacy18 is the
reproduction view; all20 and main22 are explicitly mechanical extensions.
Scores are frozen before opening released prelim truth. No main truth is read.
No submission files or remote writes are produced.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, platform, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'original'))
import 分析 as common
import 非会員を使う順位較正の検討 as calib
import base, general, tapas

BACKGROUNDS = ('starter', 'prelim', 'main', 'starter+prelim', 'starter+main',
               'prelim+main', 'starter+prelim+main')
METHODS = ('P01', 'conditional_quartile500', 'null_copula_ridge01',
           'query_pairs', 'query_triples', 'fitP01_triples25')
HOLD = general.HOLD

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def matsha(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()

def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def load_phase(root, phase):
    if phase == 'prelim':
        d = root / '公式データ/予備戦'
        own = list(d.glob('PWSCup2026_prelim_team05_*/B_self.csv'))
        attack = d / 'PWSCup2026_prelim_attack'
        ref = d / 'PWSCup2026_prelim_common/A_bg.csv'
        self_id = 5
    else:
        d = root / '公式データ'
        own = list(d.glob('PWSCup2026_final_team28_*/B_self.csv'))
        attack = d / 'PWSCup2026_final_attack'
        assert len(own) == 1, own
        ref = own[0].parent / 'A_bg.csv'
        self_id = 28
    assert len(own) == 1, own
    read = lambda p: pd.read_csv(p, float_precision='round_trip')
    paths = [own[0], ref] + sorted(attack.glob('C_*.csv')) + sorted(attack.glob('aia_challenge_*.csv'))
    b, a = read(own[0]), read(ref)
    cs = {int(p.stem.rsplit('_', 1)[1]):read(p) for p in attack.glob('C_*.csv')}
    cs.pop(self_id, None)
    qs = {int(p.stem.rsplit('_', 1)[1]):read(p) for p in attack.glob('aia_challenge_*.csv')}
    teams = sorted(cs)
    assert len(teams) == {'prelim':20, 'main':22}[phase], teams
    assert len(b) == 1049 and b.record_id.is_unique
    private = pd.MultiIndex.from_frame(b[list(common.QI7)]).isin(pd.MultiIndex.from_frame(qs[self_id][list(common.QI7)]))
    rare = b.is_rare.to_numpy(bool)
    normal = ~private & ~rare
    assert (private.sum(), rare.sum(), (private & rare).sum(), normal.sum()) == (163,24,13,875)
    return d, paths, b, a, qs, cs, teams, private, rare, normal

def tapas_ranks(raw, pair, triple, frames, q, source, normal, teams, nb):
    """Exact fit-only calibration and tie correction from original TAPAS run."""
    train = ~np.isin(source, HOLD)
    stacks = [[], [], []]
    diagnostics = []
    for j, team in enumerate(teams):
        fit = (source != team) & train
        for feature, values, dest in zip(('fitP01','pairs','triples'),
                                          (raw[:,j],pair[:,j],triple[:,j]), stacks):
            cv = calib.calibrate(values, frames, values[nb:][fit], q.loc[fit], ['prefecture','sex'], 500)
            cv = cv - .001*rankdata(values, method='average')/len(values) + .001*np.clip(tapas.rr(values, values[nb:][fit]), 0, 1)
            dest.append(tapas.rr(cv, cv[:nb][normal]))
            threshold = float(np.quantile(cv[nb:][fit], .99))
            for cohort in HOLD:
                if cohort == team or not (source == cohort).any():
                    continue
                hold = source == cohort
                diagnostics.append({'feature':feature,'team':team,'heldout_cohort':cohort,
                    'n':int(hold.sum()),'exceed_fit_p99':float(np.mean(cv[nb:][hold] > threshold))})
    return [np.column_stack(x) for x in stacks], diagnostics

def predict(raw, pair, triple, frames, source, normal, teams, nb, scopes):
    q = frames.iloc[nb:].reset_index(drop=True)
    rs, diagnostics = tapas_ranks(raw, pair, triple, frames, q, source, normal, teams, nb)
    outputs = {}
    fits = []
    for scope, indices in scopes.items():
        target_teams = [teams[j] for j in indices]
        allsource = np.r_[np.repeat(-1, nb), source]
        sc, fit = general.fit_transform(raw[:,indices], allsource, nb, normal, target_teams, frames, calib, base)
        for f in fit:
            fits.append({'inference':scope, **f})
        fr, pr, tr = [r[:,indices] for r in rs]
        fp = fr - .1*base.rivals(fr,np.median)
        ps = pr - .1*base.rivals(pr,np.median)
        ts = tr - .1*base.rivals(tr,np.median)
        sc.update(query_pairs=ps, query_triples=ts, fitP01_triples25=.75*fp+.25*ts)
        outputs[scope] = {m:s[:nb][normal].copy() for m,s in sc.items()}
    return outputs, fits, diagnostics

def original_regression(out, methods):
    checks = []
    for family in ('general','tapas'):
        baseline = json.loads((HERE/f'baseline_{family}'/'prediction_hashes.json').read_text())
        for item in baseline:
            actual = matsha(methods[item['method']])
            checks.append({'family':family,'method':item['method'],
                           'expected_sha256':item['sha256'],'actual_sha256':actual,
                           'exact_match':actual == item['sha256']})
    dump(out/'adapter_baseline_regression.json', checks)
    assert all(x['exact_match'] for x in checks), checks

def rank_agreement(predictions, teams_by_scope):
    rows=[]
    for (bg,scope), methods in predictions.items():
        ref=predictions[('main',scope)]
        for method, values in methods.items():
            for j,team in enumerate(teams_by_scope[scope]):
                s,r=values[:,j],ref[method][:,j]
                top=np.argsort(-s,kind='stable')[:20]
                rt=np.argsort(-r,kind='stable')[:20]
                boundary=np.sort(s)[-20]
                refboundary=np.sort(r)[-20]
                rows.append({'background':bg,'reference_background':'main','inference':scope,
                    'method':method,'team':team,'spearman':float(spearmanr(s,r).statistic),
                    'kendall_tau_b':float(kendalltau(s,r).statistic),
                    'top20_overlap':int(len(set(top)&set(rt))),
                    'top20_overlap_fraction':float(len(set(top)&set(rt))/20),
                    'boundary_tie_size':int((s==boundary).sum()),
                    'reference_boundary_tie_size':int((r==refboundary).sum()),
                    'mean_abs_rank_shift':float(np.mean(np.abs(rankdata(s)-rankdata(r))))})
    return pd.DataFrame(rows)

def evaluate(out, predictions, truth, teams, scopes, normal, private, rare):
    rows=[]
    for (bg,scope),methods in predictions.items():
        for method,scores in methods.items():
            confidence=base.confidence(scores,normal,private,rare)
            for col,j in enumerate(scopes[scope]):
                for population,mask in [('all_fixed',np.ones(len(normal),bool)),('normal',normal)]:
                    yy=truth[mask,j]; ss=confidence[mask,col]
                    rows.append({'background':bg,'inference':scope,'method':method,'team':teams[j],
                        'population':population,'n':int(mask.sum()),'members':int(yy.sum()),
                        **base.metrics(ss,yy)})
    by=pd.DataFrame(rows)
    by.to_csv(out/'by_target.csv',index=False,float_format='%.17g')
    old=pd.read_csv(HERE/'original/historical_calibration_by_target.csv',float_precision='round_trip')
    expected=old[old.method=='Q_prefecture_sex_shrink500_cross01'].set_index('team').mia_all
    actual=by[(by.background=='prelim')&(by.inference=='all20')&(by.method=='P01')&(by.population=='all_fixed')].set_index('team')['tpr_0.01']
    assert set(expected.index)==set(actual.index)
    err=float((actual-expected).abs().max())
    dump(out/'all20_P01_baseline_regression.json',{'max_perteam_TPR_error':err,
        'historical_mean20':float(expected.mean()),'rerun_mean20':float(actual.mean()),
        'source_sha256':sha(HERE/'original/historical_calibration_by_target.csv'),
        'note':'Original all20 P01 validated separately from legacy18 inference; truth opened after freeze'})
    assert err<1e-12,err
    summaries=[]
    for (bg,scope,population),panel in by.groupby(['background','inference','population'],sort=False):
        for view, excluded in [('all',[]),('without1_2',[1,2])]:
            g=panel[~panel.team.isin(excluded)]
            for method,h in g.groupby('method',sort=False):
                values=h.set_index('team')['tpr_0.01']
                ref=by[(by.background=='main')&(by.inference==scope)&(by.population==population)&(by.method==method)].set_index('team')['tpr_0.01'].loc[values.index]
                delta=values-ref
                summaries.append({'background':bg,'inference':scope,'evaluation':view,
                    'population':population,'method':method,'targets':len(h),
                    'mean_tpr':values.mean(),'TPR_ge010':int((values>=.1-1e-12).sum()),
                    'delta_mainBG':delta.mean(),'wins_mainBG':int((delta>1e-12).sum()),
                    'losses_mainBG':int((delta < -1e-12).sum()),'worst_delta_mainBG':delta.min(),
                    'best_delta_mainBG':delta.max()})
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False,float_format='%.17g')

def run(root, bgdir, phase, out):
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    d,paths,b,a,qs,cs,teams,private,rare,normal=load_phase(root,phase)
    scopes={'all20':list(range(len(teams))),'legacy18':[j for j,t in enumerate(teams) if t not in (1,2)]} if phase=='prelim' else {'all22':list(range(len(teams)))}
    teams_by_scope={k:[teams[j] for j in v] for k,v in scopes.items()}
    q=pd.concat([qs[t][list(common.QI7)] for t in sorted(qs)],ignore_index=True)
    source=np.concatenate([np.repeat(t,len(qs[t])) for t in sorted(qs)])
    train=~np.isin(source,HOLD)
    assert not pd.MultiIndex.from_frame(q[train]).isin(pd.MultiIndex.from_frame(q[~train])).any()
    nb=len(b);frames=pd.concat([b[list(common.QI7)],q],ignore_index=True)
    identities=np.array([f'{phase}:Q:{t}:{v}' for t in sorted(qs) for v in qs[t].challenge_row_id])
    edges={col:np.unique(np.quantile(a[col],np.arange(.1,1,.1))).tolist() for col in common.NUM7}
    protocol={'phase':phase,'backgrounds':BACKGROUNDS,'methods':METHODS,'inference_targets':teams_by_scope,
        'interpretation':'legacy18 exact reproduction; all20/all22 mechanical scope extensions',
        'general_original':general.CONFIG,'tapas_original':tapas.CONFIG,
        'bin_edges_source':f'original full {phase} A_bg; fixed across all 7 subsets',
        'TAPAS_reference_dependency':f'Every subset, including starter-only, also uses {phase} A_bg for fixed quantile preprocessing; not a pure sole-source attack',
        'pooling':'row concatenation, preserving source+record_id in companion maps; no identity joins',
        'normal_output':'original private=0; non-private rare=.4; normal=.6+.3*minrank/n',
        'rank_agreement':'normal candidates against main background, stable original B order for top20 ties',
        'fit_Q_cohorts':sorted(set(source[train].tolist())),'hold_Q_cohorts':sorted(set(source[~train].tolist())),
        'fit_Q_rows':int(train.sum()),'hold_Q_rows':int((~train).sum()),
        'fit_Q_identity_sha256':hashlib.sha256('\n'.join(identities[train]).encode()).hexdigest(),
        'hold_Q_identity_sha256':hashlib.sha256('\n'.join(identities[~train]).encode()).hexdigest(),
        'truth':'prelim released labels after predictions freeze; main labels never read',
        'parameters_tuned_for_current_truth':False}
    dump(out/'fixed_protocol.json',protocol);dump(out/'bin_edges.json',edges)
    inputs={str(p):sha(p) for p in paths}
    for bg in BACKGROUNDS:
        p=bgdir/f'{bg}.csv'; inputs[str(p)]=sha(p)
    code={p.name:sha(p) for p in (HERE/'original').glob('*.py')};code['run_bg7.py']=sha(__file__)
    dump(out/'input_sha256.json',inputs);dump(out/'code_sha256.json',code)
    xf=tapas.binned(frames,edges)
    cb={t:tapas.binned(cs[t],edges) for t in teams}
    predictions={};hashes=[];fits=[];diagnostics=[];timings=[]
    order=('prelim',)+tuple(bg for bg in BACKGROUNDS if bg!='prelim') if phase=='prelim' else BACKGROUNDS
    for bg in order:
        t0=time.time(); aa=pd.read_csv(bgdir/f'{bg}.csv',float_precision='round_trip')
        assert not aa[list(common.QI7)].isna().any().any()
        ab=tapas.binned(aa,edges)
        cache=out/f'raw_cache_{bg}.npz';meta=out/f'raw_cache_{bg}.json'
        cache_key={'background_sha256':sha(bgdir/f'{bg}.csv'),'phase_inputs':inputs,'code':code}
        if cache.exists():
            assert json.loads(meta.read_text())==cache_key,'Stale cache'
            with np.load(cache) as z: raw,pair,triple=[z[k] for k in ('raw','pair','triple')]
        else:
            raw=[];pair=[];triple=[]
            for t in teams:
                raw.append(common.core(aa,frames,cs[t],common.NUM7,(10,))[10])
                pair.append(tapas.query_score(ab,cb[t],xf,tapas.PAIRS))
                triple.append(tapas.query_score(ab,cb[t],xf,tapas.TRIPLES))
                print(phase,bg,t,round(time.time()-start,2),flush=True)
            raw,pair,triple=[np.column_stack(x) for x in (raw,pair,triple)]
            np.savez_compressed(cache,raw=raw,pair=pair,triple=triple);dump(meta,cache_key)
        outputs,fit,diag=predict(raw,pair,triple,frames,source,normal,teams,nb,scopes)
        if phase=='prelim' and bg=='prelim': original_regression(out,outputs['legacy18'])
        fits.extend([{'background':bg,**v} for v in fit]);diagnostics.extend([{'background':bg,**v} for v in diag])
        for scope,methods in outputs.items():
            predictions[(bg,scope)]=methods
            np.savez_compressed(out/f'predictions_{bg}_{scope}.npz',**methods)
            for method,score in methods.items():
                hashes.append({'background':bg,'inference':scope,'method':method,'shape':list(score.shape),'sha256':matsha(score)})
        timings.append({'background':bg,'rows':len(aa),'seconds':time.time()-t0})
    dump(out/'prediction_hashes.json',hashes);dump(out/'null_fits.json',fits)
    pd.DataFrame(diagnostics).to_csv(out/'heldout_Q.csv',index=False)
    frozen=datetime.now(timezone.utc).isoformat()
    dump(out/'freeze_record.json',{'frozen_utc':frozen,'prediction_hashes_sha256':sha(out/'prediction_hashes.json'),'code_sha256':code})
    print(phase,'ALL PREDICTIONS FROZEN',frozen,flush=True)
    ranks=rank_agreement(predictions,teams_by_scope);ranks.to_csv(out/'rank_agreement.csv',index=False,float_format='%.17g')
    ranks.groupby(['background','inference','method'],sort=False).agg(targets=('team','size'),mean_spearman=('spearman','mean'),min_spearman=('spearman','min'),mean_top20_overlap=('top20_overlap','mean'),min_top20_overlap=('top20_overlap','min')).reset_index().to_csv(out/'rank_summary.csv',index=False,float_format='%.17g')
    audit=None
    if phase=='prelim':
        truth,audit,truthpaths=base.audit_truth(d,b,private,rare,teams)
        dump(out/'truth_sha256.json',{str(p):sha(p) for p in truthpaths})
        evaluate(out,predictions,truth,teams,scopes,normal,private,rare)
    for p,h in inputs.items(): assert sha(p)==h, p
    dump(out/'run_record.json',{'phase':phase,'elapsed_seconds':time.time()-start,'frozen_utc':frozen,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'background_timings':timings,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'normal_candidates':int(normal.sum()),'private':int(private.sum()),'rare':int(rare.sum()),
        'truth_audit':audit,'input_hashes_unchanged':True,'no_submission_changes':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--background-dir',type=Path,required=True)
    p.add_argument('--phase',choices=('prelim','main'),required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.root,args.background_dir,args.phase,args.out)
