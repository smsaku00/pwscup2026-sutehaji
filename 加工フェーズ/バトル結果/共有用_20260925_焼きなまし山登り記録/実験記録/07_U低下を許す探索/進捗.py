from pathlib import Path
import json,time,sys
p=Path(__file__).resolve().parent/'状態.json'
def bar(x):
    x=min(1,max(0,x));n=int(x*30)
    return '█'*n+'░'*(30-n)+f' {x:.0%}'
while True:
    try:s=json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):s={}
    now=time.time();elapsed=s.get('elapsed_seconds',now-s.get('started_epoch',now))
    lines=['有用性を少し下げる探索 / 約20分 / 16並列','',
        '時間の目安 '+bar(elapsed/1200)+f' 経過 {elapsed/60:.1f}分',
        '工程の進捗 '+bar(s.get('phase_index',0)/6),
        '準備 → Zhang11 → Zhang04 → 別パネルAIA → MIA → 集計','',
        f'状態: {s.get("status","準備中")} / {s.get("stage","")}',
        f'現在: {s.get("base","")} / {s.get("batch_done",0)}/{s.get("batch_total",0)}',
        f'探索済み: {s.get("new_search_candidates",0):,}候補',
        f'今回の判定: {s.get("batch_outcomes",{})}']
    for x in s.get('utility_budgets',[]):lines.append(f'{x["condition"]}: U={x["U"]:.6f}, 探索AIA={x["AIA"]:.6f}')
    if s.get('mia_total'):lines.append(f'MIA確認: {s.get("mia_done",0)}/{s["mia_total"]}')
    if s.get('error'):lines.append(s['error'])
    lines+=['','表示専用。閉じても計算は続きます。時間は目安で、工程完了が終了です。']
    if sys.stdout.isatty():print('\033[2J\033[H',end='')
    print('\n'.join(lines),flush=True)
    if s.get('status') in ['complete','failed'] or '--once' in sys.argv:break
    time.sleep(2)
