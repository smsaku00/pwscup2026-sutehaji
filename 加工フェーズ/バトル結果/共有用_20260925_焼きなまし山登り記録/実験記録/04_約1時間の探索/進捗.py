"""Read-only terminal dashboard; closing it does not stop the detached calculation."""
from pathlib import Path
import json,sys,time,os
HERE=Path(__file__).resolve().parent
def bar(n,total):
    f=min(1,max(0,n/max(1,total)));k=int(30*f)
    return '█'*k+'░'*(30-k)+f' {n}/{total} ({f:.1%})'
try:
    while True:
        try:s=json.loads((HERE/'状態.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):s={}
        status=s.get('status','未開始');elapsed=max(0,time.time()-s.get('started_epoch',time.time()))
        if status in ['failed','complete']:elapsed=s.get('elapsed_seconds',elapsed)
        lines=['Zhang11 → Zhang04 の山登り探索 / 各ベース内16並列','',
            '全体の工程  '+bar(s.get('phase_index',0),6),
            '①検算 → ②Zhang11 → ③Zhang04 → ④交換削減 → ⑤別パネル・MIA → ⑥資料','',
            f'状態: {status} / {s.get("stage","")}',f'ベース: {s.get("base","—")} / 試行中: 総{s.get("swaps",0)}交換',
            '現在の候補  '+bar(s.get('batch_done',0),s.get('batch_total',1)),
            f'最良AIA: {s.get("best_AIA","—")} / U下限: {s.get("U_floor","—")}',
            f'経過: {elapsed/3600:.2f}時間 / 15時間は目安です',
            f'今回のまとまりのAIA採点回数: {s.get("attack_calls_batch",0)}',
            '判定内訳: '+str(s.get('batch_outcomes',{}))]
        if s.get('mia_total'):lines.append('MIA: '+bar(s.get('mia_done',0),s['mia_total']))
        if s.get('pid') and status=='running':
            try:os.kill(s['pid'],0)
            except ProcessLookupError:lines.append('注意: 実行プロセスが見つかりません。実行.log を確認してください。')
            except PermissionError:pass
        if status=='failed':lines.append(s.get('error',''))
        lines+=['','工程バーは時間の割合ではありません。候補数は改善状況で変わります。',
            '表示専用です。Control+Cで閉じても計算は続きます。',
            '完了後: 結果の読み方.md / 採用条件.csv']
        if sys.stdout.isatty():print('\033[2J\033[H',end='')
        print('\n'.join(lines),flush=True)
        if '--once' in sys.argv or status in ['complete','failed']:break
        time.sleep(2)
except KeyboardInterrupt:pass
