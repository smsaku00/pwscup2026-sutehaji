from common import *
import yaml
P=plan()
sys.path.insert(0,str(Path(P['kit'])/'starter/src'))
from pwscup2026.scoring.attack_score import score_aia
CFG=yaml.safe_load((Path(P['kit'])/'participant_data/scoring_config.yaml').read_text())

def truth(p):
    t=read(p['truth'])
    if not p['true_control_rare']:
        t=t.rename(columns={'paired_member_stratum':'stratum','member_true_rare_control_unknown':'is_rare'})
    return t

def metrics(pred,t,mode='raw',rare=True):
    hit=np.abs(np.asarray(pred)-t.time.to_numpy())<=.5;m=t.label.eq('member').to_numpy()
    w=np.ones(len(t)) if mode=='raw' else t.weight_163.to_numpy()
    pm=float(np.average(hit[m],weights=w[m]));pc=float(np.average(hit[~m],weights=w[~m]));r=rate(pm,pc)
    rr=None
    if rare:
        mr=m&t.is_rare.eq(1).to_numpy();cr=~m&t.is_rare.eq(1).to_numpy()
        if mode=='raw':
            res=score_aia(None,np.asarray(pred)[m],t.onset.to_numpy()[m],t.time.to_numpy()[m],None,np.asarray(pred)[~m],t.onset.to_numpy()[~m],t.time.to_numpy()[~m],CFG,
                          member_rare_mask=t.is_rare.to_numpy()[m].astype(bool),control_rare_mask=t.is_rare.to_numpy()[~m].astype(bool))
            np.testing.assert_allclose(r,res.breakdown['r_time_all'],atol=1e-13,rtol=0);rr=float(res.breakdown['r_time_rare'])
        # Weighted 387 persons are not 13 independent rare patients. No SE-deadzone shortcut.
    return dict(pm=pm,pc=pc,R_all=r,R_rare=rr,R_star=r if rr is None else max(r,rr)),hit

def old_prediction(v,a,p):
    op=int(p['id'][3:]);rec=load(OLD/'work'/f'panel_{op:02d}'/v/a['id']/'実行記録.json')
    assert sha(rec['prediction'])==rec['prediction_sha256'];return predict_file(p,rec['prediction'])

def collect(dataset,include_candidates=False):
    panels=[p for p in P['panels'] if p['dataset']==dataset]
    variants=[v for v in P['variants'] if not v['label'].startswith('BG')]
    if include_candidates:variants+=load(HERE/'search_strong/固定候補.json')['variants']
    details=[];groups=[];bgs=[]
    for p in panels:
        t=truth(p);m=t.label.eq('member').to_numpy();g=t.stratum.to_numpy()
        for a in P['attacks']:
            bp=np.array([predictions(dataset,f'BG{i:02d}',a,p) for i in range(1,11)])
            bh=np.abs(bp-t.time.to_numpy())<=.5
            for mode in ['raw','weighted']:
                bm=[metrics(pred,t,mode,p['true_control_rare'])[0] for pred in bp]
                br={k:np.mean([x[k] for x in bm]) for k in ['pm','pc','R_all','R_star']}
                bgs.append(dict(条件=dataset,パネル=p['id'],構成=p['scenario'],攻撃=a['name'],重み=mode,反復=10,
                    背景標的的中率=br['pm'],背景標的的中率最小=min(x['pm'] for x in bm),背景標的的中率最大=max(x['pm'] for x in bm),
                    背景対照的中率=br['pc'],背景全体AIA=br['R_all'],背景全体希少最大=br['R_star']))
                w=np.ones(len(t)) if mode=='raw' else t.weight_163.to_numpy()
                for v in variants:
                    pred=old_prediction(v['label'],a,p) if dataset=='old387' and v['label'] in [x['label'] for x in P['variants'][:3]] else predictions(dataset,v['label'],a,p)
                    z,h=metrics(pred,t,mode,p['true_control_rare'])
                    gains=float(np.mean(((~bh[:,m])&h[m]).sum(axis=1)));loss=float(np.mean((bh[:,m]&~h[m]).sum(axis=1)))
                    gainw=float(np.mean(np.sum(((~bh[:,m])&h[m])*w[m],axis=1)/w[m].sum()))
                    lossw=float(np.mean(np.sum((bh[:,m]&~h[m])*w[m],axis=1)/w[m].sum()))
                    np.testing.assert_allclose(gainw-lossw,z['pm']-br['pm'],atol=1e-12,rtol=0)
                    constant,_=metrics(np.full(len(t),10.),t,mode,p['true_control_rare'])
                    details.append(dict(条件=dataset,パネル=p['id'],構成=p['scenario'],加工案=v['label'],攻撃=a['name'],攻撃ID=a['id'],重み=mode,
                        標的的中人数=int(h[m].sum()),標的的中率=z['pm'],背景標的的中率=br['pm'],標的的中率の増分=z['pm']-br['pm'],
                        背景標本別増分の最小=z['pm']-max(x['pm'] for x in bm),背景標本別増分の最大=z['pm']-min(x['pm'] for x in bm),
                        追加的中人数平均=gains,的中喪失人数平均=loss,差引追加的中人数平均=gains-loss,
                        対照的中率=z['pc'],背景対照的中率=br['pc'],対照的中率の増分=z['pc']-br['pc'],
                        標的増分と対照増分の差=(z['pm']-br['pm'])-(z['pc']-br['pc']),
                        全体AIA=z['R_all'],希少AIA=z['R_rare'],全体希少最大=z['R_star'],定数全体希少最大=constant['R_star'],
                        同手法背景全体希少最大平均=br['R_star'],同手法背景との差=z['R_star']-br['R_star']))
                    for s in range(4):
                        mask=m&(g==s);hc=float(h[mask].mean());hb=float(bh[:,mask].mean())
                        groups.append(dict(条件=dataset,パネル=p['id'],構成=p['scenario'],加工案=v['label'],攻撃=a['name'],群=GROUPS[s],人数=int(mask.sum()),
                            C標的的中率=hc,背景標的的中率=hb,標的的中率の増分=hc-hb))
    df=pd.DataFrame(details);gf=pd.DataFrame(groups).drop_duplicates();bf=pd.DataFrame(bgs)
    out(HERE/f'{dataset}_手法別の増分.csv',df);out(HERE/f'{dataset}_群別の増分.csv',gf);out(HERE/f'{dataset}_同手法背景基準.csv',bf)
    panel=[]
    for (pid,c,mode),z in df.groupby(['パネル','加工案','重み'],sort=False):
        mi=z['標的的中率の増分'].idxmax();r=z.loc[mi];worst=z['全体希少最大'].max();constant=z['定数全体希少最大'].iloc[0]
        panel.append(dict(条件=dataset,パネル=pid,構成=z.構成.iloc[0],加工案=c,重み=mode,
            最大標的増分=r['標的的中率の増分'],増分最大の攻撃=r['攻撃'],その攻撃の差引追加人数=r['差引追加的中人数平均'],
            探索に未使用の攻撃の最大標的増分=z.loc[~z.攻撃ID.isin(P['search']['attacks']+P.get('search_strong',P['search'])['attacks']),'標的的中率の増分'].max(),
            AIA最大値=worst,定数込みAIA最大=max(worst,constant),
            同手法背景差の最大=z['同手法背景との差'].max(),
            群別標的増分の最大=gf[(gf.パネル==pid)&(gf.加工案==c)]['標的的中率の増分'].max()))
    pf=pd.DataFrame(panel);out(HERE/f'{dataset}_パネル別の要約.csv',pf)
    summary=[]
    for (c,mode),z in pf.groupby(['加工案','重み'],sort=False):
        row=dict(条件=dataset,加工案=c,重み=mode,パネル数=len(z))
        for col in ['最大標的増分','AIA最大値','同手法背景差の最大','群別標的増分の最大']:
            row.update({col+'_'+name:float(fn(z[col])) for name,fn in [('最小',np.min),('中央値',np.median),('最大',np.max)]})
        summary.append(row)
    sf=pd.DataFrame(summary);out(HERE/f'{dataset}_3案の比較.csv',sf)
    print(sf.to_string(index=False),flush=True)
    return df,gf,bf,pf

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--dataset',required=True);a.add_argument('--candidates',action='store_true');x=a.parse_args();collect(x.dataset,x.candidates)
