"""Read-only terminal progress display. Does not perform calculations or API calls."""
from pathlib import Path
import json, sys, time

HERE = Path(__file__).resolve().parent
BASES = ['Zhang11', 'Zhang04', 'Seo03']
STAGES = {'preparing':'候補を準備中', 'baseline':'元案の採点・照合中', 'running':'候補を採点中',
          'validating_base':'最良候補を元の攻撃プログラムと照合中', 'reporting':'資料を作成中',
          'complete':'全て完了', 'failed':'エラーで停止'}
def bar(n, total):
    width=32; k=int(width*n/total)
    return '█'*k+'░'*(width-k)+f'  {n:3d}/{total}  {n/total:6.1%}'
try:
    while True:
        try: s=json.loads((HERE/'状態.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError): s={}
        status=s.get('status','preparing'); done=s.get('completed',0); counts=s.get('per_base',{})
        lines=['450候補テスト — 3ベースを順番に、各ベース内で16並列','', '全体  '+bar(done,450),'']
        for base in BASES:
            n=counts.get(base,0)
            marker='完了' if n==150 and not (base==s.get('base') and status=='validating_base') else '処理中' if base==s.get('base') else '待機'
            lines.append(f'{base:7s}  {bar(n,150)}  {marker}')
        lines+=['', '順序：Zhang11 → Zhang04 → Seo03', '状態：'+STAGES.get(status,status)]
        elapsed=s.get('elapsed_seconds',0) if status in ['complete','failed'] else time.time()-s.get('started_epoch',time.time())
        lines.append(f'経過：{max(elapsed,0)/60:.1f}分')
        if s.get('eta_minutes') is not None and status=='running': lines.append(f'採点完了までの目安：約{s["eta_minutes"]:.1f}分（照合時間は別）')
        if s.get('latest'): lines.append('直近：'+s['latest'])
        lines+=['','この画面は表示専用です。Control+Cで閉じても計算は続きます。']
        if sys.stdout.isatty(): sys.stdout.write('\x1b[2J\x1b[H')
        print('\n'.join(lines),flush=True)
        if status in ['complete','failed'] or '--once' in sys.argv: break
        time.sleep(1)
except KeyboardInterrupt: pass
