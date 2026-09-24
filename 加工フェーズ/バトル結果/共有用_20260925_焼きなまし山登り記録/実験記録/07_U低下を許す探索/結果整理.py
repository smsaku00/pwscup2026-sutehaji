from pathlib import Path
import json,csv,collections,hashlib,time
HERE=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def out(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

d=load(HERE/'集計.json');extra=load(HERE/'追加確認結果.json')
assert load(HERE/'検証結果.json')['passed'] and extra['passed']
prior=load(HERE.parent/'公平比較_初回合格案と長時間5案_20260924/比較集計.json')
names={c['key']:c['name'] for c in prior['cases']}
counts=[];lower=[]
for base in ['Zhang11','Zhang04']:
    c=collections.Counter();complete=collections.Counter();low_complete=[]
    for f in (HERE/'results').glob(base+'-R*/結果.json'):
        if '-confirm' in f.parent.name:continue
        r=load(f);u=r['utility']['utility']
        band='元案以上' if u>=.9219987812309569-1e-12 else '0.915〜元案未満' if u>=.915-1e-12 else '0.910〜0.915未満' if u>=.910-1e-12 else '0.900〜0.910未満' if u>=.900-1e-12 else '0.900未満'
        c[band]+=1
        if r['complete']:complete[band]+=1
        if .900-1e-12<=u<.9219987812309569-1e-12:lower.append(r)
    for band in ['元案以上','0.915〜元案未満','0.910〜0.915未満','0.900〜0.910未満','0.900未満']:
        counts.append(dict(ベース=base,U帯=band,候補数=c[band],全24採点完了数=complete[band]))
out(HERE/'共有用/U帯別の探索数.csv',counts)
total=sum(x['候補数'] for x in counts)
assert total==2912
initial=load(HERE/'確認前の候補固定.json')['candidates']
byid={r['id']:r for r in initial}
brief=[]
for base in ['Zhang11','Zhang04']:
    row=next(x for x in d['comparison'] if x['ベース']==base and x['条件']=='現状維持')
    label=names.get(byid[row['候補']]['key'],row['候補'])
    brief.append(dict(ベース=base,候補名=label,分類='U維持・既存案',有用性U=row['有用性U'],探索最大AIA=row['探索最大AIA'],
        確認387人AIA=row['確認387人AIA'],元案比163人AIA悪化組数=row['AIA悪化組数'],元案比MIA悪化数=row['MIA悪化比較数']))
for row in extra['comparison']:
    brief.append(dict(ベース=row['ベース'],候補名='今回のU低下参考案',分類='U低下・今回の新規案',有用性U=row['有用性U'],探索最大AIA=row['探索AIA'],
        確認387人AIA=row['確認387人AIA'],元案比163人AIA悪化組数=row['元案比163人AIA悪化組数'],元案比MIA悪化数=row['元案比MIA悪化数']))
out(HERE/'共有用/読みやすい比較表.csv',brief)
passed=[r for r in brief if r['元案比163人AIA悪化組数']==0 and r['元案比MIA悪化数']==0]
lines=['# Uを少し下げる探索の結果','',
    '**今回の2,912候補では、Uの下限を0.900まで緩めても、U維持の既存案より低い探索AIAの候補は見つかりませんでした。**','',
    '- Zhang11：1,408候補。Zhang04：1,504候補。16並列で順番に計算しました。',
    '- Uの下限は元案と同じ約0.922、0.915、0.910、0.900の4条件。両ベースとも4条件の最良は同じ既存案です。',
    f'- U維持条件では除外される **{len(lower)}候補**（0.900以上・元案未満）も、今回はAIA評価に進めました。',
    '- 新しい確認パネルは387標的1組と163標的3組。各組に同数の対照を置きました。MIAは26標的＋1,023対照の3組です。','',
    '| 候補 | 実測U | 探索AIA | 新しい387人組のAIA | 163人AIA悪化組数 | MIA悪化比較数 |',
    '|---|---:|---:|---:|---:|---:|']
for r in brief:lines.append(f'| {r["候補名"]} | {r["有用性U"]:.6f} | {r["探索最大AIA"]:.6f} | {r["確認387人AIA"]:.6f} | {r["元案比163人AIA悪化組数"]}/3 | {r["元案比MIA悪化数"]}/12 |')
lines+=['','- Uは高い方、AIAは低い方が良い値です。悪化数は、それぞれの元のZhang11・Zhang04からの変化を数えています。',
    '- 探索用と確認用は違うデータです。両列の値を直接比べて改善率を計算しません。',
    '- 「今回のU低下参考案」は、実際にUが下がり、探索24採点を通過した候補の中で探索AIAが最小の案です。追加確認の抽出基準は確認結果を見る前に固定しました。',
    '- Zhang04では、Uが元案を下回った候補のうち、全24採点まで進んで親案のAIAを改善したものはありませんでした。','',
    '## どう判断するか','',
    '- **今回の1交換ずつの山登りでは、Uの下限を緩めるだけによる最良AIAの更新はありませんでした。**',
    '- 現行探索は「交換のたびにAIAが改善すること」も要求しています。この制約が探索の行き詰まりに関わる可能性がありますが、今回だけで主因は断定できません。',
    '- 次の検討候補は、複数の交換をまとめて1案として評価する方式です。今回は実施していません。',
    '- 今回のパネルで悪化がなくても、過去のパネルでの悪化がなくなったことにはなりません。採用判断では過去の公平比較も併せて見ます。',
    '- 公式の希少MIA・本番匿名性得点は未評価。φ補正も適用していません。','',
    '## 詳細を見る順番','',
    '1. `読みやすい比較表.csv`：既存案とU低下案を比較。',
    '2. `U帯別の探索数.csv`：どの程度Uが下がる候補を試したか。',
    '3. `条件別比較.csv`：4つのU条件の判定。',
    '4. `U低下案_追加確認.csv`、`U低下案_AIAパネル別.csv`：追加確認の詳細。','',
    '根拠：本戦Bによるローカル模擬実験。有用性は公式コードによる自己採点。']
(HERE/'共有用/00_結果まとめ.md').write_text('\n'.join(lines)+'\n')
for f,h in load(HERE/'計画.json')['hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
report=dict(passed=True,new_candidates=total,relaxed_candidates_evaluated=len(lower),summary=brief,
    elapsed_seconds=extra['ended_epoch']-load(HERE/'状態.json')['started_epoch'],frozen_inputs_unchanged=True)
(HERE/'最終集計検証.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
