from pathlib import Path
import numpy,pandas,joblib,json
import runtime as r
from joblib import Memory
u=r.initialize();data=r.data_for(u,'starter+prelim+main');a=u.base.read(Path(r.MANIFEST['subsets']['starter+prelim+main']['path']));b,_,q,p,rare,cs,targets,own=u.base.phase_inputs(r.ROOT,'prelim')
f=Memory('/workspace/shared/mia_recovered_calibration/cache/legacy_read_only_hashing',verbose=0).cache(u.base.core)
root=Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/legacy/cache/分析/core')
for name,aa,bb,cc in [('fresh',a,b,cs[7]),('drop',data['a'].drop(columns='age_bin'),data['b'].drop(columns='age_bin'),data['cs'][7]),('fresh_copy',a.copy(),b.copy(),cs[7].copy())]:
 k=f._get_args_id(aa,bb,cc,tuple(u.base.NUM7),(1,3,10),False);print(name,k,(root/k/'output.pkl').exists(),flush=True)
print(numpy.__version__,pandas.__version__,joblib.__version__)
