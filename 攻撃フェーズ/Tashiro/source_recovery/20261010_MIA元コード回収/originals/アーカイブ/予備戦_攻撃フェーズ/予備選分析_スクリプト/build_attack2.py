# -*- coding: utf-8 -*-
"""提出 #2『指紋・構造・採点式逆算』型。最近傍MIA / DOMIAS密度比 とは別系統。

 MIA: (a) 2種類の A_bg を両方除外した完全一致指紋だけを確定行にする
      (b) C_k が A_bg の逐語コピーの標的では確定行を置かない（FPR予算を焼かない）
      (c) 東京都=非共有ブロックの構造事前確率 + 交差標的ダブルセンタリング残差
 AIA: (d) 採点式 R=(p_m-p_c)/(1-p_c) の逆算。既定は打ち切り定数 9.52（p=0.8834）で
          p_c を最大に釘付けし、分母 (1-p_c)=0.117 を 8.5倍の増幅器として使う
      (e) 丸め許容の完全一致で拾えた行だけ、C_k の time が『早期イベント』を主張する
          ときに限り上書きする（打ち切り行は定数のままの方が強い）
"""
import argparse, shutil, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/Users/taisei1105/pws2026-local/work')
from teams import SHORT

ROOT  = Path('/Users/taisei1105/pws2026-local')
ATK   = ROOT/'予備戦/配布データ/PWSCup2026_prelim_attack'
TOKEN = ROOT/'work/data_prelim_team05/token.txt'
SELF  = 5
CENS  = 9.52          # 窓[9.02,10.02]。9.499と同じ p=0.8834 で浮動小数の余裕がある
EARLY = 9.02          # C_k の time がこれ未満なら『早期イベント』の主張とみなして上書き
LAB   = ['BMI','SBP','TG','HDL','ALT','FPG']
MQI   = ['age','SBP','TG','HDL','ALT']          # challenge が持つ連続列
R9    = lambda x: np.round(np.asarray(x, float), 9)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--round',type=int,default=1)
    ap.add_argument('--out',default='prelim_attack/r2'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    B  = pd.read_csv(ROOT/'share_team05/data/B.csv')
    A1 = pd.read_csv(ROOT/'share_team05/data/A_bg.csv')
    A2 = pd.read_csv(ROOT/'スターターキット/初版/participant_data/A_bg.csv')
    A  = pd.concat([A1,A2],ignore_index=True)
    # 攻撃者全員が持てる公開プール。★ share_team05/B_practice.csv は自Bの複製なので入れてはいけない
    PUBF=[ROOT/'share_team05/data/A_bg.csv',
          ROOT/'スターターキット/初版/participant_data/A_bg.csv',
          ROOT/'スターターキット/初版/participant_data/B_practice.csv',
          ROOT/'スターターキット/改訂版_20260824/participant_data/B_practice.csv']
    PUB=[pd.read_csv(f) for f in PUBF]
    rare = pd.read_csv(ROOT/'share_team05/data/B_self.csv')['is_rare'].to_numpy(bool)
    rid  = B['record_id'].to_numpy(int)
    tokyo= (B['prefecture']=='東京都').to_numpy()
    targets=[k for k in sorted(int(p.stem.split('_')[1]) for p in ATK.glob('C_*.csv')) if k!=SELF]

    # ---------- (a) 完全一致指紋。公開プール由来の対照で Fisher 検定して門を作る ----------
    from scipy.stats import fisher_exact
    CONT=LAB+['time']
    pub={c:set().union(*[set(R9(d[c])) for d in PUB]) for c in CONT}
    bset={c:set(R9(B[c])) for c in CONT}
    uniq={}; ctlv={}
    for c in CONT:
        m={}
        for i,v in enumerate(R9(B[c])): m.setdefault(v,[]).append(i)
        uniq[c]={v:L[0] for v,L in m.items() if len(L)==1 and v not in pub[c]}
        pm={}
        for d in PUB:
            for v in R9(d[c]): pm[v]=pm.get(v,0)+1
        ctlv[c]={v for v,n in pm.items() if n==1 and v not in bset[c]}

    # ---------- (b) C_k が A_bg の逐語コピーか ----------
    akey=set(map(tuple,R9(A[LAB].to_numpy(float))))

    # ---------- 距離行列（残差スコア用。確定行の判定には使わない） ----------
    mu=A[LAB].mean().to_numpy(float); sd=A[LAB].std(ddof=0).to_numpy(float); sd=np.where(sd>0,sd,1.)
    Bz=(B[LAB].to_numpy(float)-mu)/sd
    from sklearn.neighbors import NearestNeighbors
    Dmat=np.zeros((len(B),len(targets))); conf={}; copycat={}; aia=[]; ninfo={}; nfp={}; fisher={}
    # 帰無用の対照: 自Bの各行と (sex,prefecture,smoking,onset,death) 一致・age±1 の A_bg 行
    KEY=['sex','prefecture','smoking','onset','death']; aidx={}
    for jj,(kk,ag) in enumerate(zip(map(tuple,A[KEY].to_numpy(object)),A['age'].to_numpy())):
        aidx.setdefault(kk,[]).append((ag,jj))
    _r=np.random.default_rng(0); ctl=[]
    for _s in range(6):
        js=[]
        for kk,ag in zip(map(tuple,B[KEY].to_numpy(object)),B['age'].to_numpy()):
            cc=[jj for aa,jj in aidx.get(kk,[]) if abs(aa-ag)<=1]
            if cc: js.append(_r.choice(cc))
        ctl.append(np.array(js))

    # challenge 照合用のスケール
    qmu=A[MQI].mean().to_numpy(float); qsd=A[MQI].std(ddof=0).to_numpy(float); qsd=np.where(qsd>0,qsd,1.)

    for ti,k in enumerate(targets):
        C=pd.read_csv(ATK/f'C_{k}.csv')
        Dmat[:,ti]=NearestNeighbors(n_neighbors=1).fit((C[LAB].to_numpy(float)-mu)/sd).kneighbors(Bz)[0][:,0]
        ck=list(map(tuple,R9(C[LAB].to_numpy(float))))
        copycat[k]=np.mean([t in akey for t in ck])
        s=set(); nfp_k=0; fpp={}
        if copycat[k] < 0.95:                       # 逐語コピーの標的では確定行を置かない
            ALPHA=0.01/(len(CONT)*len(targets))     # Bonferroni（140 比較）
            for c in CONT:
                cv=set(R9(C[c]))
                h={uniq[c][v] for v in cv & set(uniq[c])}
                if not h: continue
                nh,nc_=len(h),len(ctlv[c]&cv)
                # 公開プールの一意値を対照にした Fisher 片側。偶然の水準なら採らない
                pv=(fisher_exact([[nh,len(uniq[c])-nh],[nc_,len(ctlv[c])-nc_]],alternative='greater')[1]
                    if len(ctlv[c])>=20 and len(uniq[c])>=20 else 1.0)
                fpp[c]=(nh,nc_,pv)
                if pv<ALPHA: s|=h
            nfp_k=len(s); fisher[k]=fpp
            # 帰無0が確認できた近接一致だけ足す（対照A_bg行が1行も入らない閾値）
            nnB=NearestNeighbors(n_neighbors=1).fit((C[LAB].to_numpy(float)-mu)/sd)
            nl=np.concatenate([nnB.kneighbors((A.iloc[p_][LAB].to_numpy(float)-mu)/sd)[0][:,0] for p_ in ctl])
            for t_ in (0.15,0.05):
                if (nl<t_).sum()==0:
                    s |= set(np.flatnonzero(Dmat[:,ti]<t_).tolist()); break
        conf[k]=s; nfp[k]=nfp_k

        # ---------- (d)(e) AIA ----------
        Q=pd.read_csv(ATK/f'aia_challenge_{k}.csv')
        th=np.full(len(Q),CENS)
        Cz=(C[MQI].to_numpy(float)-qmu)/qsd; Qz=(Q[MQI].to_numpy(float)-qmu)/qsd
        d,j=NearestNeighbors(n_neighbors=1).fit(Cz).kneighbors(Qz)
        d=d[:,0]; j=j[:,0]
        # 谷を自動検出する。素材を残した標的だけ低距離の山が分離するので、そこだけ採る
        srt=np.sort(d); g=int(np.argmax(np.diff(srt[:200])))
        # 谷が本物と言えるのは (i) 跳びが十分大きい (ii) 山が十分低い (iii) 山に10行以上ある
        # (iv) 相手が A_bg の逐語コピーでない、の4条件がそろうときだけ
        th_gate=(srt[g]+srt[g+1])/2 if ((srt[g+1]-srt[g])>0.02 and srt[g]<0.20
                                        and g+1>=10 and copycat[k]<0.95) else -1.0
        # age と sex の完全一致を独立の裏取りとして要求する
        hit=(d<th_gate)&(C['age'].to_numpy()[j]==Q['age'].to_numpy())&(C['sex'].to_numpy()[j]==Q['sex'].to_numpy())
        ct=C['time'].to_numpy(float)[j]
        ov=hit&(ct<EARLY)                            # 早期イベントの主張のときだけ上書き
        th[ov]=ct[ov]
        ninfo[k]=(int(hit.sum()),int(ov.sum()))
        aia.append(pd.DataFrame({'target_k':k,'challenge_row_id':Q['challenge_row_id'].to_numpy(int),'time_hat':th}))

    # ---------- (c) 帯構成 ----------
    L=-np.log(np.clip(Dmat,1e-12,None))
    resid=L-L.mean(1,keepdims=True)-L.mean(0,keepdims=True)+L.mean()
    rare_nt=np.flatnonzero(rare&~tokyo); rare_all=np.flatnonzero(rare)
    F=pd.DataFrame({'record_id':rid})
    for ti,k in enumerate(targets):
        c=np.zeros(len(B)); r=resid[:,ti]; rk=pd.Series(r).rank(pct=True).to_numpy()
        c[tokyo]  = 0.02+0.08*rk[tokyo]                       # 帯5 非共有ブロック側
        c[~tokyo] = 0.12+0.38*rk[~tokyo]                      # 帯4
        cand=np.flatnonzero(~tokyo&~rare)
        nb=max(4,10-len(conf[k]))   # FPR1%の予算(≈10行)から確定行ぶんを差し引く
        top=cand[np.argsort(-r[cand])[:nb]]
        if len(top): c[top]=0.70+0.10*np.linspace(1,0,len(top))   # 帯2 予算の残り
        pick=rare_nt[(ti+a.round)%len(rare_nt)]               # 帯3 希少ラテン方格
        order=[pick]+[x for x in rare_nt if x!=pick]+[x for x in rare_all if tokyo[x]]
        for pos,ix in enumerate(order): c[ix]=0.58-0.06*pos/max(len(order)-1,1)
        for ix in conf[k]: c[ix]=1.0                          # 帯1 指紋の確定行
        F[str(k)]=np.clip(c,0.,1.)

    F.to_csv(out/'F_mia.csv',index=False)
    pd.concat(aia,ignore_index=True).to_csv(out/'F_aia.csv',index=False)
    shutil.copy(TOKEN,out/'token.txt')
    print(f"F_mia {F.shape[0]}行×{F.shape[1]-1}標的 / F_aia {sum(len(x) for x in aia)}行 → {out}\n")
    print(f"{'k':>3} {'略称':<12}{'A_bg逐語率':>10}{'指紋':>6}{'近接':>6}{'帯1計':>7}{'AIA一致':>8}{'AIA上書き':>9}")
    for k in targets:
        mark=' ← 情報なし' if copycat[k]>=0.95 else ''
        print(f"{k:3d} {SHORT[k]:<12}{copycat[k]:10.3f}{nfp[k]:6d}{len(conf[k])-nfp[k]:6d}"
              f"{len(conf[k]):7d}{ninfo[k][0]:8d}{ninfo[k][1]:9d}{mark}")
    print("\n■ 指紋チャネルの検定（公開プールの一意値を対照にした Fisher 片側）")
    for k in targets:
        for c,(nh,nc_,pv) in sorted(fisher.get(k,{}).items()):
            print(f"   k={k:2d} {SHORT[k]:<12} {c:<5} 標的{nh:3d}  対照{nc_:4d}  p={pv:.3f}"
                  +("  ★採用" if pv<0.01/(len(CONT)*len(targets)) else "  → 偶然の水準。不採用"))
    print(f"\n  AIA 既定定数 = {CENS}（窓 [{CENS-0.5:.2f}, {CENS+0.5:.2f}]、p_m=0.883）")
    print( "  上書きは『C_k の time < 9.02 = 早期イベントの主張』の行だけ。打ち切り行は定数のままが強い")

if __name__=='__main__': main()
