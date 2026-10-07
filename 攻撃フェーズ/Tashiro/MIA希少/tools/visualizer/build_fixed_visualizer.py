"""Build the newly verified, nonadaptive five-question teaching page."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / 'docs' / 'visualizer'


def main():
    code_path = DEST / 'nonadaptive_code.json'
    encoding_path = ROOT / 'analysis/full_depth4_20261007/universal_score_encoding.json'
    padding_path = ROOT / 'analysis/full_depth4_20261007/padding_rule.json'
    code = json.loads(code_path.read_text(encoding='utf-8'))
    encoding = json.loads(encoding_path.read_text(encoding='utf-8'))
    padding = json.loads(padding_path.read_text(encoding='utf-8'))
    if code['candidate_ids'] != padding['candidate_ids'] or code['total_high'] != encoding['total_high']:
        raise ValueError('fixed code disagrees with shared score settings')
    ids = code['candidate_ids']
    queries = code['query_masks']
    if len(queries) != 5 or any(bin(query).count('1') != 5 for query in queries):
        raise ValueError('expected five fixed sets of five candidates')
    expected = [sum(1 << i for i, word in enumerate(code['codewords']) if word >> r & 1) for r in range(5)]
    if expected != queries:
        raise ValueError('codeword bits disagree with fixed questions')
    data = {'ids': ids, 'query_masks': queries, 'codewords': code['codewords'],
            'teams': code['team_order'], 'caps': code['caps'], 'high': code['total_high'],
            'negative_ids': padding['known_negative_ids'], 'padding_counts': code['padding_counts'],
            'encoding': encoding, 'lesson_defaults': code['demo_membership_masks'],
            'demo_masks': code['joint_demo_membership_masks'], 'provenance': code['provenance']}
    matrix = '<table><thead><tr><th>ID</th>' + ''.join(f'<th>質問{r+1}</th>' for r in range(5)) + '</tr></thead><tbody>'
    for person, word in zip(ids, code['codewords']):
        matrix += f'<tr><td>{person}</td>' + ''.join(f'<td class="code-{ "one" if word >> r & 1 else "zero" }">{word >> r & 1}</td>' for r in range(5)) + '</tr>'
    matrix += '</tbody></table>'
    padding_table = '<table><thead><tr><th>Team</th><th>候補の選択数</th><th>追加する既知非所属ID</th><th>高側合計</th></tr></thead><tbody>'
    for team, count, high in zip(code['team_order'], code['padding_counts'], code['total_high']):
        padding_table += f'<tr><td>{team}</td><td>5人</td><td>' + '・'.join(map(str, padding['known_negative_ids'][:count])) + f'（{count}人）</td><td>{high}人</td></tr>'
    padding_table += '</tbody></table>'
    inputs = '<table class="fixed-input-grid"><thead><tr><th>質問</th><th>調べる候補ID</th><th>希少平均点</th><th>復号</th></tr></thead><tbody>'
    for r, query in enumerate(queries):
        selected = '・'.join(str(person) for i, person in enumerate(ids) if query >> i & 1)
        inputs += f'<tr data-fixed-round="{r+1}" data-query="{query}"><td>{r+1}</td><td>{selected}</td><td><input id="fixed-mean-{r+1}" aria-label="質問{r+1}の平均点" placeholder="mia.tpr_rareの全桁"></td><td><button id="fixed-decode-{r+1}">この行を復号</button></td></tr><tr><td colspan="4" id="fixed-answer-{r+1}" class="small">未入力</td></tr>'
    inputs += '</tbody></table>'
    style = re.search(r'<style>([\s\S]*?)</style>', (HERE / 'viewer5.template.html').read_text(encoding='utf-8')).group(1)
    ui = "'use strict';\nconst data=JSON.parse(document.getElementById('fixed-data').textContent),five=MIA_FIXED,shared=MIA_TREE,ids=data.ids,pop=five.popcount;\n"
    ui += (HERE / 'fixed_lesson_ui.js').read_text(encoding='utf-8')
    ui += '\n' + (HERE / 'fixed_operations_ui.js').read_text(encoding='utf-8')
    replacements = {'STYLE': style, 'SHARED_CORE': (HERE / 'tree_core.js').read_text(encoding='utf-8'),
                    'FIXED_CORE': (HERE / 'fixed_core.js').read_text(encoding='utf-8'),
                    'DATA': json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'),
                    'FIXED_UI': ui, 'CODE_MATRIX': matrix, 'PADDING_TABLE': padding_table, 'MEAN_INPUTS': inputs}
    html = (HERE / 'fixed5.template.html').read_text(encoding='utf-8')
    for key, value in replacements.items():
        html = html.replace(f'/*__{key}__*/', value)
    if re.search(r'/\*__[A-Z_]+__\*/', html):
        raise ValueError('unreplaced fixed-page placeholder')
    output = DEST / 'MIA_5query_cards.html'
    output.write_text(html, encoding='utf-8')
    report = {'strategy': 'nonadaptive', 'html_bytes': output.stat().st_size,
              'html_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'format': 'single offline HTML with five unchanged questions',
              'single_team_states': [11, 55], 'single_team_max_depth': [5, 5],
              'guarantee_method': code['guarantee']['basis'], 'provenance': code['provenance'],
              'source_files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                               'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in (code_path, encoding_path, padding_path)]}
    (DEST / 'size_report_5query.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'source_files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
