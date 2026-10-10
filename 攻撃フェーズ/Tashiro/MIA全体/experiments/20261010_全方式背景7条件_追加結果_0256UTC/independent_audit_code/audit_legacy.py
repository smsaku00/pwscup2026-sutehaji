"""Independent aggregate-score audit; never imports experiment scoring functions."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = Path(__file__).resolve().parent
L = Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/legacy')
D = Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
read = lambda p: pd.read_csv(p, float_precision='round_trip')
b = read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')))
q = read(D/'PWSCup2026_prelim_attack/aia_challenge_5.csv')
qi = ['age', 'sex', 'prefecture', 'SBP', 'TG', 'HDL', 'ALT']
private = pd.MultiIndex.from_frame(b[qi]).isin(pd.MultiIndex.from_frame(q[qi]))
rare = b.is_rare.to_numpy(bool)
normal = ~private & ~rare
pools = {int(p.stem.rsplit('_',1)[1]): read(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')}
assert np.array_equal(pools[5].record_id, b.record_id)
ys = {t: pools[5].pool_id.isin(v.pool_id).to_numpy(int) for t, v in pools.items()}

def score(s, y, alpha=.01):
    order = np.argsort(-s, kind='stable')
    ss, yy = s[order], y[order]
    ends = np.r_[np.where(np.diff(ss) != 0)[0], len(s)-1]
    tp = np.r_[0, np.cumsum(yy)[ends]] / y.sum()
    fp = np.r_[0, (np.arange(len(s))+1-np.cumsum(yy))[ends]] / (len(y)-y.sum())
    return float(np.interp(alpha, fp, tp)), float(np.trapezoid(tp, fp))

reports, errors, auc_errors = [], [], []
hashes = samples = 0
for p in sorted((L/'frozen/prelim').glob('*/*/manifest.json')):
    if not (p.parent/'by_target.csv').exists():
        continue
    meta = json.loads(p.read_text())
    table = read(p.parent/'by_target.csv').set_index(['team', 'method'])
    assert not table.index.duplicated().any()
    assert len(meta['teams']) == 20
    assert meta['evaluation_labels_read_after_prediction_freeze'] is True
    if meta['family'] == 'logistic':
        assert meta['training_labels_used'] is True
        assert meta['training_label_protocol'] == 'person5_target4_joint_exclusion_OOF'
    with np.load(p.parent/'matrices.npz') as bank:
        assert set(bank.files) == set(meta['methods']) == set(meta['prediction_sha256'])
        for name in bank.files:
            matrix = bank[name]
            assert matrix.shape == (875,20) and np.isfinite(matrix).all()
            assert hashlib.sha256(np.ascontiguousarray(matrix, dtype='<f8').tobytes()).hexdigest() == meta['prediction_sha256'][name]
            hashes += 1
            for j in [0,19]:
                team = meta['teams'][j]
                conf = np.zeros(1049)
                conf[rare & ~private] = .4
                conf[normal] = .6 + .3 * rankdata(matrix[:,j], method='min') / 875
                tpr, auc = score(conf, ys[team])
                row = table.loc[(team,name)]
                errors.append(abs(tpr - row.mia_all))
                auc_errors.append(abs(auc - row.auc))
                samples += 1
    reports.append({'background':meta['background'], 'family':meta['family'], 'configurations':len(meta['methods']), 'target_rows':len(table), 'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

assert max(errors, default=0) < 1e-12
assert max(auc_errors, default=0) < 1e-12
status = json.loads((L/'aggregate/status.json').read_text()) if (L/'aggregate/status.json').exists() else {}
report = {
    'status': 'passed_current_completed_files',
    'primary_complete': status.get('primary_completed') == 1750,
    'method': 'Independent tie-group ROC and trapezoid AUC; first/last target for every frozen configuration; all matrix hashes.',
    'prediction_hashes_checked': hashes, 'target_rows_recomputed': samples,
    'max_abs_TPR_error': max(errors,default=0), 'max_abs_AUC_error':max(auc_errors,default=0),
    'population': 'all_fixed1049: known-private confidence0; rare unknown confidence0.4; normal unknown875 ranked0.6..0.9',
    'scope': reports,
    'label_policy': 'OOF training reads preliminary labels before prediction freeze; evaluation reads labels after freeze. No main truth.',
}
(R/'inventory/legacy_independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='scope'},ensure_ascii=False))
