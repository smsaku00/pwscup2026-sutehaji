"""Read-only progress screen. No model calls; closing this screen leaves the job running."""
from pathlib import Path
import json,time,sys
HERE=Path(__file__).resolve().parent
def bar(frac):
    f=max(0,min(1,frac));k=int(30*f)
    return '█'*k+'░'*(30-k)+f' {f:6.1%}'
def stamp(value):return time.strftime('%m/%d %H:%M',time.localtime(value)) if value else '準備中'
try:
    while True:
        try:s=json.loads((HERE/'状態.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):s={}
        now=time.time();start=s.get('started_epoch',now);target=s.get('target_end_epoch',start+15*3600)
        status=s.get('status','準備中');elapsed=max(0,now-start)
        if status in ['complete','failed']:elapsed=s.get('elapsed_seconds',elapsed)
        done=s.get('batch_done',0);total=s.get('batch_total',1)
        lines=['約15時間の継続探索 / Zhang11 → Zhang04 / 各16並列','',
            '時間の目安  '+bar(elapsed/max(1,target-start)),
            f'経過 {elapsed/3600:.2f}時間 / 全工程終了の目安 {stamp(target)}',
            f'探索終了の目安 {stamp(s.get("search_end_epoch"))}（その後、別パネルとMIAの確認）','',
            '全体の工程  '+bar(s.get('phase_index',0)/6),
            '検算 → Zhang11 → Zhang04 → 交換削減 → 別パネル・MIA → 資料','',
            f'状態: {status} / {s.get("stage","")}',
            f'ベース: {s.get("base","—")} / 第{s.get("round",0)}組 / 総交換数: {s.get("swaps","—")}',
            '今回の候補  '+bar(done/max(1,total))+f'  {done:,}/{total:,}',
            f'今回の実行で探索済み: {s.get("new_search_candidates",0):,}候補（完了した組の合計）',
            f'ベース別: {s.get("per_base_candidates",{})}',
            f'前回と同じ候補の除外用に登録: {s.get("previous_search_candidates",0):,}件（現在のベース）',
            f'最良AIA: {s.get("best_AIA","—")} / U下限: {s.get("U_floor","—")}',
            f'改善なし: {s.get("no_gain_rounds",0)}組連続 → 回数では止めません',
            f'このベースの区切り: {stamp(s.get("base_deadline_epoch"))}',
            f'今回の判定: {s.get("batch_outcomes",{})}']
        if s.get('mia_total'):lines.append(f'MIA {s.get("mia_done",0)}/{s["mia_total"]}')
        if status=='failed':lines.append(s.get('error',''))
        lines+=['','時間になったら現在の候補のまとまりを終えて切り替えます。',
            '時間バーは配分の目安、工程バーは工程の完了率です。',
            '表示専用です。Control+Cで閉じても計算は続きます。']
        if sys.stdout.isatty():print('\033[2J\033[H',end='')
        print('\n'.join(lines),flush=True)
        if status in ['complete','failed'] or '--once' in sys.argv:break
        time.sleep(2)
except KeyboardInterrupt:pass
