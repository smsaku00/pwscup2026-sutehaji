"""Build compact and detailed offline lessons for the fixed five-investigation code."""
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
    ids, queries = code['candidate_ids'], code['query_masks']
    if len(queries) != 5 or any(bin(query).count('1') != 5 for query in queries):
        raise ValueError('expected five fixed sets of five candidates')
    expected = [sum(1 << i for i, word in enumerate(code['codewords']) if word >> r & 1) for r in range(5)]
    if expected != queries:
        raise ValueError('codeword bits disagree with fixed investigations')
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
    prefix = "'use strict';\nconst data=JSON.parse(document.getElementById('fixed-data').textContent),five=MIA_FIXED,shared=MIA_TREE,ids=data.ids,pop=five.popcount;\n"
    operations = (HERE / 'fixed_operations_ui.js').read_text(encoding='utf-8')
    shared = {'SHARED_CORE': (HERE / 'tree_core.js').read_text(encoding='utf-8'),
              'FIXED_CORE': (HERE / 'fixed_core.js').read_text(encoding='utf-8'),
              'DATA': json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'),
              'PADDING_TABLE': padding_table}
    cards = ''.join(f'<article id="glass-card-{i+1}" class="glass-card" style="--layer:{i+1};--z:{i+2}"><div id="glass-meta-{i+1}" class="card-meta"></div><div id="glass-grid-{i+1}" class="sheet-grid"></div></article>' for i in range(5))
    legacy_style = re.search(r'<style>([\s\S]*?)</style>', (HERE / 'viewer5.template.html').read_text(encoding='utf-8')).group(1)
    configurations = [
        ('cards', 'fixed_records.template.html', {'STYLE': (HERE / 'fixed_compact.css').read_text(encoding='utf-8') + '\n' + (HERE / 'fixed_records.css').read_text(encoding='utf-8'),
         'FIXED_UI': prefix + (HERE / 'fixed_lesson_ui.js').read_text(encoding='utf-8') + '\n' + operations.replace('調査ID', '対象ID').replace('質問', '提出') + '\n' + (HERE / 'fixed_overview_ui.js').read_text(encoding='utf-8'),
         'CODE_MATRIX': matrix.replace('質問', '提出'), 'MEAN_INPUTS': inputs.replace('質問', '提出'), 'GLASS_CARDS': cards},
         'size_report_5query.json', 'single offline HTML; eleven record patterns with one/two selected overlays, integer sums and a six-Team response table'),
        ('pairs', 'fixed_compact.template.html', {'STYLE': (HERE / 'fixed_compact.css').read_text(encoding='utf-8'),
         'FIXED_UI': prefix + (HERE / 'fixed_pairs_lesson_ui.js').read_text(encoding='utf-8') + '\n' + operations.replace('調査ID', '対象ID').replace('質問', '提出') + '\n' + (HERE / 'fixed_pairs_overview_ui.js').read_text(encoding='utf-8'),
         'CODE_MATRIX': matrix.replace('質問', '提出'), 'MEAN_INPUTS': inputs.replace('質問', '提出'), 'GLASS_CARDS': cards},
         'size_report_5query_pairs.json', 'preserved offline HTML; six-Team response table and five transparent sheets with 55 pair cells'),
        ('reference', 'fixed_reference.template.html', {'STYLE': legacy_style,
         'FIXED_UI': prefix + (HERE / 'fixed_reference_ui.js').read_text(encoding='utf-8') + '\n' + operations,
         'CODE_MATRIX': matrix, 'MEAN_INPUTS': inputs},
         'size_report_5query_reference.json', 'single offline HTML with five unchanged questions'),
    ]
    for name, template, changes, report_name, format_name in configurations:
        html = (HERE / template).read_text(encoding='utf-8')
        for key, value in {**shared, **changes}.items():
            html = html.replace(f'/*__{key}__*/', value)
        if re.search(r'/\*__[A-Z_]+__\*/', html):
            raise ValueError('unreplaced fixed-page placeholder')
        output = DEST / f'MIA_5query_{name}.html'
        output.write_text(html, encoding='utf-8')
        report = {'strategy': 'nonadaptive', 'html_bytes': output.stat().st_size,
                  'html_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'format': format_name,
                  'single_team_states': [11, 55], 'single_team_max_depth': [5, 5],
                  'guarantee_method': code['guarantee']['basis'], 'provenance': code['provenance'],
                  'source_files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                  for p in (code_path, encoding_path, padding_path)]}
        (DEST / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'file': output.name, 'bytes': report['html_bytes'], 'sha256': report['html_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
