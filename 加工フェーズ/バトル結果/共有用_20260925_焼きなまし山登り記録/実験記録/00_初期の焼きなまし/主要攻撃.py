"""The original additional attacks, reused without labels, with background training cached."""
from common import *
from 評価 import metrics,truth

class Attack:
    def __init__(self,base,need_background=True):
        self.plan=plan();self.config=self.plan['search_strong']
        self.atks={a['id']:a for a in self.plan['attacks']};self.panels=[p for p in self.plan['panels'] if p['id'] in self.config['panels']]
        self.mods={i:mod('strong_'+i,RUN/self.atks[i]['path']) for i in self.config['attacks']}
        self.models={i:m.Attack(m.read_table(self.plan['a_bg'],'background')) for i,m in self.mods.items()}
        self.q={i:[m.read_table(p['challenge'],'challenge') for p in self.panels] for i,m in self.mods.items()}
        self.truth=[truth(p) for p in self.panels];self.baseline={}
        if need_background:
            for aid in self.config['attacks']:
                self.baseline[aid]=[]
                for p,t in zip(self.panels,self.truth):
                    preds=[predictions(p['dataset'],f'BG{i:02d}',self.atks[aid],p) for i in range(1,11)]
                    zs=[metrics(pr,t,'weighted',p['true_control_rare'])[0] for pr in preds]
                    hit=np.abs(np.array(preds)-t.time.to_numpy())<=.5;m=t.label.eq('member').to_numpy()
                    self.baseline[aid].append(dict(pm=float(np.mean([z['pm'] for z in zs])),R=float(np.mean([z['R_all'] for z in zs])),
                        groups=[float(hit[:,m&t.stratum.eq(s).to_numpy()].mean()) for s in range(4)]))
    def predictions(self,c):
        tc=c[COLS].copy()
        for col in COLS:
            if col!='prefecture':tc[col]=pd.to_numeric(tc[col].map(lambda x:format(x,'.17g')))
        for col in ['age','sex','smoking','onset','death']:tc[col]=tc[col].astype(np.int64)
        tc=tc.sort_values(COLS,kind='stable').reset_index(drop=True)
        result={}
        for aid,model in self.models.items():
            m=self.mods[aid];ps=[]
            for q in self.q[aid]:
                pr=model.predict(tc,q)[0];pr,_=m.exact_override(q,tc,np.asarray(pr,float));ps.append(pr)
            result[aid]=ps
        return result
    def evaluate(self,c):
        preds=self.predictions(c);rows=[];all_lift=[];group_lift=[];rlift=[]
        for aid,prs in preds.items():
            for j,(pr,t) in enumerate(zip(prs,self.truth)):
                z,h=metrics(pr,t,'weighted',self.panels[j]['true_control_rare']);b=self.baseline[aid][j];m=t.label.eq('member').to_numpy()
                lift=z['pm']-b['pm'];gl=[float(h[m&t.stratum.eq(s).to_numpy()].mean())-b['groups'][s] for s in range(4)]
                dl=z['R_all']-b['R'];all_lift.append(max(0,lift));group_lift.extend(max(0,x) for x in gl);rlift.append(max(0,dl))
                rows.append(dict(attack=aid,panel=self.panels[j]['id'],weighted_member_lift=lift,group_lifts=gl,weighted_R_lift=dl))
        energy=max(all_lift)+.25*max(group_lift)+.25*max(rlift)+.1*float(np.mean(all_lift))
        return dict(energy=energy,member_lift_max=max(all_lift),group_lift_max=max(group_lift),R_lift_max=max(rlift),records=rows)
