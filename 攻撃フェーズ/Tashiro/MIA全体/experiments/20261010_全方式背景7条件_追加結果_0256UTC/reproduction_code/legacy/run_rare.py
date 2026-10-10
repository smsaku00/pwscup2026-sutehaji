"""Rerun the original three rare-distance diagnostics without changing rarity roles."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,hashlib,types,fcntl
import pandas as pd,numpy as np
BASE=Path(__file__).resolve().parent;SRC=BASE/'source_rare'
sys.path.insert(0,str(SRC))
import 分析 as rare
import 共通設定 as support
import レポート as report
p=argparse.ArgumentParser();p.add_argument('--background',required=True);args=p.parse_args()
root=Path('/workspace/shared/mia_bg7_inputs/root/公式データ');attack=root/'PWSCup2026_final_attack';own=next(root.glob('PWSCup2026_final_team28_*'))
bg=Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(args.background+'.csv');out=BASE/'rare_results'/args.background
(BASE/'locks').mkdir(exist_ok=True)
lock=(BASE/'locks'/('rare__'+args.background+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'completion.json').exists():sys.exit(0)
original=support.checked_paths
# Verify every original distributed file against original pinned hashes first.
opt=types.SimpleNamespace(attack_dir=attack,b_self=own/'B_self.csv',background=own/'A_bg.csv',output_dir=out)
paths,manifest=original(opt)
original_bg_hash=manifest['sha256']['A_bg.csv'];new_bg_hash=rare.sha(bg)
selected_manifest=json.loads(Path('/workspace/shared/mia_bg7_inputs/manifest.json').read_text())
assert bg.is_file() and len(pd.read_csv(bg)) in (5143,10286,15429)
assert new_bg_hash==selected_manifest['subsets'][args.background]['sha256']
def checked(options):
 verified,old=original(opt)
 assert rare.sha(bg)==new_bg_hash
 verified['A_bg.csv']=bg
 return verified,{**old,'selected_background':args.background,'selected_background_sha256':new_bg_hash,'original_background_sha256':original_bg_hash}
rare.checked_paths=checked
# Historical qualitative candidate selections are not refitted by these distance methods.
# Preserve them under an explicit historical label; do not regenerate old narrative as new findings.
report.write_reports=lambda frames,run,output:None
sys.argv=[str(SRC/'分析.py'),'--attack-dir',str(attack),'--b-self',str(own/'B_self.csv'),'--background',str(bg),'--output-dir',str(out)]
start=time.perf_counter();rare.main()
old_selection=out/'集計/チーム別候補.csv'
if old_selection.exists():old_selection.rename(out/'集計/旧候補選定に新距離を付記.csv')
record={'background':args.background,'seconds':time.perf_counter()-start,'diagnostic_settings':list(rare.METHODS),'targets':22,'rare_rows':24,'candidate_rows':11,'known_nonmember_rows':13,'main_truth_used':False,'unknown_pairs':242,'selected_background_sha256':new_bg_hash,'source_hashes':{p.name:rare.sha(p) for p in SRC.iterdir() if p.is_file()},'narrative_regenerated':False}
(out/'completion.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False))
