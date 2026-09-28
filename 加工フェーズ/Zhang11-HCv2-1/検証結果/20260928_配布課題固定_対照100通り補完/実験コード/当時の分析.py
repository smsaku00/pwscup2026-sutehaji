"""配布課題の公開QIを固定し、対照の未知値を条件付き補完する感度分析。"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREVIOUS = ROOT/'本戦/攻撃フェーズ/Tashiro/20260928_自チーム28_AIA12手法検証'
KIT = ROOT/'本戦/配布データ/pwscup2026-kit-main_revised0912'
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
NUM = QI[:5]
PHI = .089047
SEED = 2026092801
N = 100
CONFIGS = [('主条件', 40, 0), ('近傍15人', 15, 0), ('近傍80人', 80, 0), ('年齢10歳帯も固定', 40, 10)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def out(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')


def say(message):
    print(time.strftime('[%H:%M:%S] ')+message, flush=True)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def freeze(source, dest):
    source, dest = Path(source), Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert sha(source) == sha(dest), dest
    else:
        shutil.copyfile(source, dest)
    return str(dest)


def prepare():
    meta = load(PREVIOUS/'実験条件.json')
    for filename, digest in meta['frozen_hashes'].items():
        assert sha(filename) == digest, filename
    assert sha(PREVIOUS/'分析.py') == meta['analysis_sha256']
    assert load(PREVIOUS/'回帰検証.json')['passed']
    assert load(PREVIOUS/'採点検証.json')['passed']
    predictor = freeze(PREVIOUS/'分析.py', HERE/'programs/既存予測器.py')
    old = load(PREVIOUS/'予測設定.json'); jobs = []
    for name, c_path in [('Zhang11', ROOT/'本戦/バトル/0925/7案_追加20組比較/C/C01.csv'), ('Zhang11-HCv2-1', Path(old['C']))]:
        job = dict(old, id=name)
        job['C'] = freeze(c_path, HERE/'public'/f'{name}.csv')
        job['a_bg'] = freeze(old['a_bg'], HERE/'public/A_bg.csv')
        job['panel'] = {'id':'official_team28', 'challenge':freeze(old['panel']['challenge'], HERE/'public/aia_challenge_28.csv')}
        job['helper'] = freeze(old['helper'], HERE/'programs/凍結予測共通.py')
        job['attacks'] = []
        for a in old['attacks']:
            a = dict(a); src = Path(a['local_path'])
            for f in src.parent.glob('*.py'):
                freeze(f, HERE/'programs'/a['id']/f.name)
            a['local_path'] = str(HERE/'programs'/a['id']/src.name)
            job['attacks'].append(a)
        settings = HERE/'settings'/f'{name}.json'; save(settings, job); jobs.append((name, settings))
    own = list((ROOT/'本戦').glob('配布*/PWSCup2026_final_team28_*/B_self.csv')); assert len(own) == 1
    freeze(own[0], HERE/'評価用非公開/B_self.csv')
    freeze(KIT/'participant_data/scoring_config.yaml', HERE/'settings/scoring_config.yaml')
    previous_root = ROOT/'予備戦/配布データ'
    truth_dir = next((previous_root/'結果').glob('正解*'))
    previous_data = []
    for truth in sorted(truth_dir.glob('aia_truth_*.csv')):
        qp = previous_root/'PWSCup2026_prelim_attack'/truth.name.replace('truth', 'challenge')
        if not qp.exists():
            continue
        tcopy = freeze(truth, HERE/'評価用非公開/予備戦'/truth.name)
        qcopy = freeze(qp, HERE/'評価用非公開/予備戦'/qp.name)
        d = read(qcopy).merge(read(tcopy), on='challenge_row_id', validate='one_to_one', sort=False)
        d['team'] = int(truth.stem.split('_')[-1]); previous_data.append(d)
    prior = pd.concat(previous_data, ignore_index=True)
    assert len(prior) == 21*326
    for (_, label), d in prior.groupby(['team', 'label']):
        assert len(d) == 163 and d.is_rare.sum() == 13 and d.onset.sum() == 16
        assert d.loc[d.is_rare.astype(bool), 'onset'].sum() == 4
    out(HERE/'評価用非公開/予備戦結合.csv', prior)
    records = {str(f.relative_to(HERE)):sha(f) for root in ['public','programs','settings','評価用非公開'] for f in (HERE/root).rglob('*') if f.is_file()}
    save(HERE/'計画.json', {'created_before_scores':True, 'seed':SEED, 'n':N, 'phi':PHI,
        'configs':CONFIGS, 'primary':'主条件', 'input_hashes':records, 'analysis_sha256':sha(__file__),
        'prediction':'公開326行をそのまま入力。2案×12手法を再予測。補完値・Bを攻撃に渡さない。',
        'imputation':'予備戦の真の対照3423行から近傍条件付き補完。C・攻撃予測は補完モデルに使わない。',
        'constraints':'希少13・発症16・希少発症4。性別・県ごとの発症人数を標的と一致。',
        'age_band_note':'予備戦21組中5歳帯では3組・10歳帯では1組で標的対照の人数不一致。全組一致する県・性別を主条件とし、年齢は近傍距離に含める。本戦28では人数が一致する10歳帯を追加した感度確認も行う。',
        'validation':'予備戦21チームを1つずつ全行除外し、その対照の真値を隠して100通り補完。',
        'interpretation':'モデルを仮定した仮想正解。主条件100組の範囲は公式得点の信頼区間ではない。',
        'python':sys.version, 'pandas':pd.__version__, 'numpy':np.__version__})
    return jobs, predictor, prior


def quota_draw(prob, count, rng):
    """独立Bernoulliを合計countで条件付けした正確な標本抽出。"""
    p = np.clip(np.asarray(prob, float), 1e-6, 1-1e-6); n = len(p); count = int(count)
    assert 0 <= count <= n
    if count == 0: return np.zeros(n, bool)
    if count == n: return np.ones(n, bool)
    f = np.zeros((n+1, count+1)); f[0,0] = 1
    for i, pi in enumerate(p, 1):
        f[i] = f[i-1]*(1-pi); f[i,1:] += f[i-1,:-1]*pi
    assert f[n,count] > 0
    result = np.zeros(n, bool); k = count
    for i in range(n, 0, -1):
        if not k: break
        chance = p[i-1]*f[i-1,k-1]/f[i,k]
        if rng.random() < chance: result[i-1] = True; k -= 1
    assert k == 0 and result.sum() == count
    return result


def group_keys(frame, width):
    bands = (frame.age//width).astype(int) if width else np.zeros(len(frame),int)
    return list(zip(frame.prefecture, frame.sex.astype(int), bands))


class Imputer:
    def __init__(self, donors, query, quota_members, k=40, width=0):
        self.donors = donors.reset_index(drop=True); self.q = query.reset_index(drop=True)
        self.k = k; self.width = width
        a = donors[NUM].to_numpy(float).copy(); b = query[NUM].to_numpy(float).copy()
        for column in [2,4]: a[:,column] = np.log1p(a[:,column]); b[:,column] = np.log1p(b[:,column])
        scale = np.maximum(np.std(a, axis=0, ddof=1), 1e-6)
        distances = cdist(b/scale, a/scale, metric='sqeuclidean')
        for category in ['sex','prefecture']:
            distances += 9*(query[category].to_numpy()[:,None] != donors[category].to_numpy()[None,:])
        self.types = (2*donors.is_rare.astype(int)+donors.onset.astype(int)).to_numpy()
        indices = np.argsort(distances, axis=1, kind='stable')[:,:min(k,len(donors))]
        joint = np.zeros((len(query),4)); prior = np.bincount(self.types,minlength=4)/len(donors)
        for row, ix in enumerate(indices):
            w = self.weights(distances[row,ix]); joint[row] = .95*np.bincount(self.types[ix],weights=w,minlength=4)+.05*prior
        self.joint = joint
        qk, mk = group_keys(query,width), group_keys(quota_members,width)
        self.groups = []
        for key in sorted(set(qk)):
            rows = np.array([i for i,x in enumerate(qk) if x==key]); ref = quota_members.iloc[[i for i,x in enumerate(mk) if x==key]]
            assert len(rows) == len(ref), (width,key,len(rows),len(ref))
            self.groups.append((rows,int(ref.onset.sum())))
        self.rare_counts = {event:int(((quota_members.is_rare==1)&(quota_members.onset==event)).sum()) for event in [0,1]}
        assert self.rare_counts == {0:9,1:4}
        self.choices = {}
        for typ in range(4):
            valid = np.flatnonzero(self.types==typ); assert len(valid)>0
            for row in range(len(query)):
                ix = valid[np.argsort(distances[row,valid],kind='stable')[:min(k,len(valid))]]
                self.choices[row,typ] = (ix,self.weights(distances[row,ix]))

    @staticmethod
    def weights(d):
        # 選んだ近傍の遠さに応じた指数重み。絶対距離と近傍幅は別。
        bandwidth = max(float(np.median(d)),1e-6)
        w = np.exp(-.5*(d-d.min())/bandwidth)
        return w/w.sum()

    def draw(self, rng):
        event = np.zeros(len(self.q),int); rare = np.zeros(len(self.q),int)
        event_prob = self.joint[:,1]+self.joint[:,3]
        for rows,count in self.groups: event[rows] = quota_draw(event_prob[rows],count,rng)
        for onset in [0,1]:
            rows = np.flatnonzero(event==onset)
            prob = self.joint[rows,2+onset]/self.joint[rows][:,[onset,2+onset]].sum(axis=1)
            rare[rows] = quota_draw(prob,self.rare_counts[onset],rng)
        donor_rows = []
        for row,typ in enumerate(rare*2+event):
            ix,w = self.choices[row,int(typ)]; donor_rows.append(rng.choice(ix,p=w))
        sampled = self.donors.iloc[donor_rows]
        assert event.sum()==16 and rare.sum()==13 and event[rare.astype(bool)].sum()==4
        return {'time':sampled.time.to_numpy(float),'onset':event,'is_rare':rare,'donor_row':np.array(donor_rows)}


def validate_model(prior):
    donors = prior[prior.label=='control'].reset_index(drop=True)
    output = []
    for index,team in enumerate(sorted(prior.team.unique()),1):
        test = donors[donors.team==team].reset_index(drop=True)
        train = donors[donors.team!=team].reset_index(drop=True)
        members = prior[(prior.team==team)&(prior.label=='member')].reset_index(drop=True)
        imputer = Imputer(train,test,members,40,0); rng = np.random.default_rng(SEED+int(team))
        draws = [imputer.draw(rng) for _ in range(N)]
        times = np.array([d['time'] for d in draws]); rare = np.array([d['is_rare'] for d in draws],bool)
        counts = (times>=9.5).sum(axis=1); rare_counts = ((times>=9.5)&rare).sum(axis=1)
        actual = int((test.time>=9.5).sum()); actual_rare = int(((test.time>=9.5)&test.is_rare.astype(bool)).sum())
        lo,hi = np.quantile(counts,[.05,.95]); rlo,rhi = np.quantile(rare_counts,[.05,.95])
        prob = (times>=9.5).mean(axis=0); rp = rare.mean(axis=0)
        output.append({'team':int(team),'n':len(test),'long_actual':actual,'long_mean':float(counts.mean()),'long_p05':float(lo),'long_p95':float(hi),'long_covered':bool(lo<=actual<=hi),
            'rare_long_actual':actual_rare,'rare_long_mean':float(rare_counts.mean()),'rare_long_p05':float(rlo),'rare_long_p95':float(rhi),'rare_long_covered':bool(rlo<=actual_rare<=rhi),
            'long_brier':float(np.mean((prob-(test.time>=9.5).to_numpy())**2)),
            'rare_brier':float(np.mean((rp-test.is_rare.to_numpy())**2))})
        say(f'補完モデル検証 [{index:02}/21] team{team}: 10年回答の的中 真値{actual} / 補完平均{counts.mean():.1f}')
    out(HERE/'集計/予備戦による補完検証.csv',output)
    return output


def get_truth():
    q = read(HERE/'public/aia_challenge_28.csv'); b = read(HERE/'評価用非公開/B_self.csv')
    merged = q.merge(b[QI+['time','onset','is_rare']],on=QI,how='left',validate='one_to_one',sort=False)
    assert merged.challenge_row_id.tolist()==q.challenge_row_id.tolist()
    member = merged.time.notna().to_numpy(); assert member.sum()==163 and (~member).sum()==163
    m = merged[member].reset_index(drop=True); c = merged[~member].reset_index(drop=True)
    assert m.is_rare.sum()==13 and m.onset.sum()==16
    out(HERE/'評価用非公開/実標的と未知対照の照合.csv',merged)
    return q,m,c,member


def evaluate(prior, jobs):
    sys.path.insert(0,str(KIT/'starter/src'))
    from pwscup2026.scoring.attack_score import score_aia
    cfg = yaml.safe_load((HERE/'settings/scoring_config.yaml').read_text()); assert cfg['scoring']['aia_rare_se_z']==1.64
    q,m,c,member = get_truth(); donors = prior[prior.label=='control'].reset_index(drop=True)
    assert not pd.concat([donors[QI],c[QI]]).duplicated(QI).any()
    predictions = {}
    for name,_ in jobs:
        with np.load(HERE/'予測'/name/'予測.npz') as z: predictions[name] = {aid:z[aid].copy() for aid in z.files}
    # 現行Cの再予測が保存済みの配布課題12手法と同じかを確認。
    with np.load(PREVIOUS/'予測/予測.npz') as z:
        for aid,values in predictions['Zhang11-HCv2-1'].items(): np.testing.assert_array_equal(values,z[aid])
    attack_ids = ['A'+str(i).zfill(2) for i in range(1,13)]
    detailed, maxima, controls = [],[],[]
    for setting,k,width in CONFIGS:
        imputer = Imputer(donors,c,m,k,width); rng = np.random.default_rng(SEED)
        for iteration in range(N):
            truth = imputer.draw(rng)
            private = c[QI+['challenge_row_id']].copy()
            for col in ['time','onset','is_rare']: private[col] = truth[col]
            private['truth_status'] = '仮想補完・本番の正解ではない'
            if setting=='主条件':
                private['donor_team'] = donors.iloc[truth['donor_row']].team.to_numpy()
                private['donor_challenge_row_id'] = donors.iloc[truth['donor_row']].challenge_row_id.to_numpy()
                out(HERE/'評価用非公開/対照の仮想正解100組'/f'{iteration+1:03}.csv',private)
            controls.append({'setting':setting,'iteration':iteration+1,'long_n':int((truth['time']>=9.5).sum()),'rare_long_n':int(((truth['time']>=9.5)&truth['is_rare'].astype(bool)).sum())})
            for name, pred in predictions.items():
                scores = {}
                for aid,values in pred.items():
                    r = score_aia(None,values[member],m.onset.to_numpy(int),m.time.to_numpy(float),None,values[~member],truth['onset'],truth['time'],cfg,
                        member_rare_mask=m.is_rare.to_numpy(bool),control_rare_mask=truth['is_rare'].astype(bool))
                    raw = float(r.score); corrected = max(0,raw-PHI); scores[aid] = raw
                    # 独立した式で公式関数の値を照合。
                    hit_m = np.abs(values[member]-m.time.to_numpy())<=.5; hit_c = np.abs(values[~member]-truth['time'])<=.5
                    rm,rc = m.is_rare.to_numpy(bool),truth['is_rare'].astype(bool)
                    pm,pc = hit_m.mean(),hit_c.mean(); pr,cr = hit_m[rm].mean(),hit_c[rc].mean()
                    all_score = np.clip((pm-pc)/max(1-pc,1e-9),0,1)
                    se = np.sqrt(pr*(1-pr)/rm.sum()+cr*(1-cr)/rc.sum())
                    rare_score = np.clip((pr-cr-1.64*se)/max(1-cr,1e-9),0,1)
                    assert abs(raw-max(all_score,rare_score))<1e-12
                    detailed.append({'setting':setting,'iteration':iteration+1,'candidate':name,'attack':aid,'raw_AIA':raw,'phi_adjusted_AIA':corrected,
                        'R_all':float(all_score),'R_rare':float(rare_score),'p_member':float(pm),'p_control':float(pc),'p_member_rare':float(pr),'p_control_rare':float(cr)})
                raw = max(scores[aid] for aid in attack_ids); maximum = max(0,raw-PHI)
                maxima.append({'setting':setting,'iteration':iteration+1,'candidate':name,'raw_max12':raw,'phi_adjusted_max12':maximum,'AIA_only_protection':1-maximum,
                    'max_attacks':'/'.join(aid for aid in attack_ids if abs(scores[aid]-raw)<1e-12),
                    'constant_raw':scores['B01'],'constant_phi':max(0,scores['B01']-PHI),'background_raw':scores['B02'],'background_phi':max(0,scores['B02']-PHI)})
            if (iteration+1)%20==0: say(f'{setting} [ {iteration+1:03}/100 ] 2案×12攻撃＋定数・背景を採点')
    out(HERE/'集計/全手法_全補完ケース.csv',detailed)
    out(HERE/'集計/12攻撃最大値_全補完ケース.csv',maxima)
    out(HERE/'集計/補完した対照の構成.csv',controls)
    return pd.DataFrame(detailed),pd.DataFrame(maxima)


def summarize(values):
    a = np.asarray(values,float)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),'p05':float(np.quantile(a,.05)), 'p95':float(np.quantile(a,.95)), 'positive_n':int((a>1e-12).sum())}


def report(detailed,maxima,validation):
    rows=[]; paired=[]
    for (setting,name),g in maxima.groupby(['setting','candidate'],sort=False):
        for metric in ['raw_max12','phi_adjusted_max12','AIA_only_protection']:
            rows.append({'setting':setting,'candidate':name,'metric':metric,**summarize(g[metric])})
    out(HERE/'集計/候補別集計.csv',rows)
    methods=[]
    names = {a['id']:a['name'] for a in load(HERE/'settings/Zhang11-HCv2-1.json')['attacks']}|{'B01':'全員10年','B02':'背景のみ1NN'}
    for (setting,name,aid),g in detailed.groupby(['setting','candidate','attack'],sort=False):
        for metric in ['raw_AIA','phi_adjusted_AIA']:
            methods.append({'setting':setting,'candidate':name,'attack':aid,'method':names[aid],'metric':metric,**summarize(g[metric])})
    out(HERE/'集計/手法別集計.csv',methods)
    for setting,g in maxima.groupby('setting',sort=False):
        wide=g.pivot(index='iteration',columns='candidate',values='phi_adjusted_max12')
        diff=wide['Zhang11-HCv2-1']-wide['Zhang11']
        paired.append({'setting':setting,'better':int((diff< -1e-12).sum()),'tie':int((abs(diff)<=1e-12).sum()),'worse':int((diff>1e-12).sum()),'difference_after_minus_before':float(diff.mean())})
    out(HERE/'集計/交換前後の対応比較.csv',paired)
    results={'summary':rows,'paired':paired,'validation':validation,'methods':methods}
    save(HERE/'比較結果.json',results)
    primary={r['candidate']:r for r in rows if r['setting']=='主条件' and r['metric']=='phi_adjusted_max12'}
    coverage=sum(r['long_covered'] for r in validation); rarecoverage=sum(r['rare_long_covered'] for r in validation)
    text='''# 配布課題を固定した対照100通りの補完評価

## 前提

- **本番の正解を復元したものではありません。** 配布された自チーム28の326行を固定し、未知の対照163人について仮想のtime・onset・希少ラベルを100通り補完した感度分析です。
- 標的163人はB_selfとの公開7項目の一意な完全一致で照合し、実際のtimeと希少ラベルを使います。公開QI・行順・C・攻撃の設定は固定しています。
- 交換前のZhang11と、配布C_28（Zhang11-HCv2-1）を同じ仮想正解で比較します。有用性Uは両案とも既存の公式値0.9219987812309569です。
- 対照の補完にC・攻撃予測・攻撃スコアは使いません。補完結果を攻撃に渡さず、全326行に対して12手法を2案とも再予測しました。

## 仮説

- 予備戦で公開された真の対照3423人の、公開属性と追跡年数・希少性との関係が本戦にもある程度当てはまると仮定します。
- 100通りの幅はこの仮定の下でのばらつきです。本番得点の信頼区間や、未知の攻撃を含む保証ではありません。

## 分析方法

- 主条件：公開5数値（TG・ALTはlog1p）を標準化した距離に、性別・県の不一致ペナルティを加え、40近傍から条件付き補完します。
- 希少13人、発症16人、希少発症4人を各組で固定。標的と対照の性別・県ごとの発症人数も一致させます。年齢は近傍距離に含めます。予備戦には年齢帯の人数が厳密一致しない組があるため主条件で帯別人数を強制せず、本戦28で人数一致する10歳帯を追加した場合も別途検証します。
- onset、希少ラベルを人数条件付きBernoulliで抽出し、割り当てたonset・希少区分に該当する近傍からtimeを抽出します。公開QIは書き換えません。元の対照人物を置き換える処理ではありません。
- 主条件100通りに加え、15近傍・80近傍・10歳帯の人数も固定した条件で各100通りの感度確認を行いました。補完方法は採点前に固定しました。
- 公式コードを使い、全体のRと、希少側のz=1.64の誤差補正を入れたRの最大を取ります。12手法の最大から **φ=0.089047** を引き、0未満を0とします。以前の「定数と背景1NNの高い方を引く」値とは違います。
- 定数回答・背景1NNも同じ仮想正解で別途採点。MIAは今回の対照補完では評価できないため、最終匿名性・総合点は確定しません。

```text
各手法のR = max(全体R, 希少Rの誤差補正後)
AIA失点 = max(0, 12手法の最大R − 0.089047)
AIAだけを見た保護値 = 1 − AIA失点
本番の匿名性 = 1 − max(MIA成分, 全攻撃者のAIA失点)
```

## 結論

### 主条件100通り：φ差引後のAIA失点（低いほど良い）

| 候補 | 平均 | 中央値 | 最小 | 最大 | 5〜95%範囲 | 0より大きい組 |
|---|---:|---:|---:|---:|---:|---:|
'''
    for name,r in primary.items(): text+=f"| {name} | {r['mean']:.6f} | {r['median']:.6f} | {r['min']:.6f} | {r['max']:.6f} | {r['p05']:.6f}〜{r['p95']:.6f} | {r['positive_n']}/100 |\n"
    comparison=next(x for x in paired if x['setting']=='主条件')
    text+=f"\n- 交換後が低い **{comparison['better']}組**、同点 **{comparison['tie']}組**、高い **{comparison['worse']}組**。平均差（交換後−交換前）は **{comparison['difference_after_minus_before']:.6f}** です。\n"
    text+='\n### 補完方法を変えたとき（交換後、各100通り）\n\n| 条件 | 平均AIA失点 | 中央値 | 最大 | 5〜95%範囲 |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        if r['candidate']=='Zhang11-HCv2-1' and r['metric']=='phi_adjusted_max12':
            text+=f"| {r['setting']} | {r['mean']:.6f} | {r['median']:.6f} | {r['max']:.6f} | {r['p05']:.6f}〜{r['p95']:.6f} |\n"
    text+='\n### 手法別（交換後・主条件100通り）\n\n| 手法 | φ差引後の平均 | 中央値 | 最大 |\n|---|---:|---:|---:|\n'
    for r in sorted(methods,key=lambda x:(-x['mean'],x['attack'])):
        if r['setting']=='主条件' and r['candidate']=='Zhang11-HCv2-1' and r['metric']=='phi_adjusted_AIA':
            text+=f"| {r['attack']} {r['method']} | {r['mean']:.6f} | {r['median']:.6f} | {r['max']:.6f} |\n"
    text+=f'''\n### 補完の妥当性と限界

- 予備戦で1チームずつ対照の真値を隠し、残り20チームだけから100通り補完しました。10年回答の的中人数の5〜95%範囲に真値が入ったのは **{coverage}/21チーム**、希少側は **{rarecoverage}/21チーム** でした。これは集団の一部の指標の検証で、個人ごとの真値を当てた保証ではありません。
- 対照の未知値の結び付き・死亡や打切りの分布・予備戦と本戦の差にモデル誤差が残ります。100通りを増やしても、この仮定の誤りは解消しません。
- 実際の163標的は固定したため、以前の20組・300組の「標的を抽出し直す評価」とは直接比較できません。
- AIAだけの保護値が高くても、MIAと他チームの未知の攻撃による失点は別です。最終匿名性を確定したとは扱いません。

### ファイルと再現

- [候補別集計](集計/候補別集計.csv)、[手法別集計](集計/手法別集計.csv)、[全ケース](集計/12攻撃最大値_全補完ケース.csv)、[予備戦での補完検証](集計/予備戦による補完検証.csv)。
- `評価用非公開/対照の仮想正解100組/001.csv`〜`100.csv`：163人分の仮想time・onset・希少ラベル。必ず仮想補完として扱います。
- `評価用非公開/` はB由来の照合情報も含むため、自チーム外向けの共有物には入れません。
- ルートから `.venv/bin/python -B 本戦/バトル/0928/配布課題固定_対照100通り補完/分析.py`。乱数seedと条件は `計画.json` に固定。
- 根拠：公式ルールブックmain-process-20260912 §4.2・§6.2.2、公式 `score_aia`、今回の配布課題・C、予備戦公開正解。本番の対照の正解そのものは未公開です。
'''
    (HERE/'説明.md').write_text(text)


def main():
    started=time.monotonic(); jobs,predictor,prior=prepare(); say('入力・補完条件を固定しました')
    for name,settings in jobs:
        with (HERE/(name+'_予測.log')).open('w') as log:
            subprocess.run([sys.executable,'-B',predictor,'--predict',str(settings),'--output',str(HERE/'予測'/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
        say(name+'：配布326行の12手法＋2基準を再予測済み')
    validation=validate_model(prior)
    detailed,maxima=evaluate(prior,jobs)
    report(detailed,maxima,validation)
    plan=load(HERE/'計画.json')
    for rel,digest in plan['input_hashes'].items(): assert sha(HERE/rel)==digest,rel
    assert len(list((HERE/'評価用非公開/対照の仮想正解100組').glob('*.csv')))==100
    save(HERE/'検証結果.json',{'passed':True,'main_imputations':100,'sensitivity_imputations':300,'preliminary_holdouts':21,
        'official_score_matches_independent_formula':True,'current_C_predictions_match_saved':True,'input_hashes_unchanged':True,
        'seconds':time.monotonic()-started,'analysis_sha256':sha(__file__)})
    say(f'完了: {time.monotonic()-started:.1f}秒。説明.md・CSV・仮想正解100組を保存。')


if __name__=='__main__': main()
