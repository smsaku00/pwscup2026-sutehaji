"""P01と既存ALT除外Gen-LRAの固定背景感度。回答・個人配列は保存しない。"""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import background_common as bc

HERE=Path(__file__).resolve().parent
SETTINGS=[
    ('Q_prefecture_sex_shrink500_cross01',bc.u.NUM7,'county500',.1),
    ('drop_ALT_cross01',tuple(x for x in bc.u.NUM7 if x!='ALT'),'raw',.1),
]

def main():
    started=time.perf_counter();data=bc.setup();predictions={}
    spec=[dict(method=n,attributes=list(cols)+['sex','prefecture'],calibration=cal,cross=cross,
               k=10,rho=.98,TG_ALT='log1p',scale='A IQR/1.349',training=False,
               fixed_before_background_test=True) for n,cols,cal,cross in SETTINGS]
    with threadpool_limits(limits=1):
        for label,changed in bc.conditions(data):
            banks={n:{} for n,_,_,_ in SETTINGS}
            for i,t in enumerate(data['teams'],1):
                for n,cols,cal,cross in SETTINGS:
                    raw=bc.u.base.core(changed['a'],changed['query'],changed['cs'][t],cols,(10,))[10]
                    banks[n][t]=bc.variant(raw,changed,t,cal)
                if i%4==0:bc.log(stage='gen_prediction',condition=label,targets_complete=i)
            for n,_,_,cross in SETTINGS:predictions[label,n]=bc.cross(banks[n],changed,cross)
            bc.log(stage='gen_condition_complete',condition=label,background_rows=len(changed['a']))
    bc.score_and_save(data,predictions,'gen_results.csv','gen_run.json',spec,started,
        extra={'own_code_sha256':bc.sha(__file__),
               'controls':'foreign Q 6520; own-target Q excluded; same Q for each background',
               'fixed_before_predictions':['Q_prefecture_sex_shrink500_cross01','drop_ALT_cross01']})

if __name__=='__main__':main()
