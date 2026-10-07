"""Build a standalone five-query explanation from the saved official policy."""
import hashlib
import json
from pathlib import Path
from artifact_layout import artifact_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'docs' / '検証資料' / '追加5回の全候補確認.json'
DEST = ROOT / 'docs' / 'visualizer'


def clean(node):
    if 'member_mask' in node:
        return {'member_mask': node['member_mask']}
    return {'query_mask': node['query_mask'], 'before': node['before'],
            'children': {h: clean(child) for h, child in node['children'].items()}}


def main():
    policy = json.loads(PROOF.read_text(encoding='utf-8'))
    encoding_path = ROOT / 'analysis/full_depth4_20261007/universal_score_encoding.json'
    padding_path = ROOT / 'analysis/full_depth4_20261007/padding_rule.json'
    encoding = json.loads(encoding_path.read_text(encoding='utf-8'))
    padding = json.loads(padding_path.read_text(encoding='utf-8'))
    if policy['total_high'] != encoding['total_high'] or policy['candidate_ids'] != padding['candidate_ids']:
        raise ValueError('five-query model does not match shared encoding')
    data = {'ids': policy['candidate_ids'], 'teams': policy['team_order'],
            'caps': policy['caps'], 'high': policy['total_high'],
            'negative_ids': padding['known_negative_ids'],
            'policies': {cap: clean(node) for cap, node in policy['policies'].items()},
            'encoding': encoding,
            'stats': {cap: {k: v for k, v in stats.items() if k != 'trace_AK'}
                      for cap, stats in policy['single_team'].items()},
            'demo_masks': [64, 1025, 2, 260, 512, 24],
            'demo_scope': '保存済みホワイトボード例の仮の所属を、検証済み5回案と固定高側合計で再計算した閲覧用経路。実際の所属結果ではありません。'}
    template = (HERE / 'viewer5.template.html').read_text(encoding='utf-8')
    core = (HERE / 'tree_core.js').read_text(encoding='utf-8')
    five = (HERE / 'five_core.js').read_text(encoding='utf-8')
    lesson = (HERE / 'lesson_ui.js').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = template.replace('/*__SHARED_CORE__*/', core).replace('/*__FIVE_CORE__*/', five).replace('/*__DATA__*/', payload).replace('/*__LESSON_UI__*/', lesson)
    if any(p in html for p in ['/*__SHARED_CORE__*/', '/*__FIVE_CORE__*/', '/*__DATA__*/', '/*__LESSON_UI__*/']):
        raise ValueError('missing source placeholder')
    DEST.mkdir(exist_ok=True)
    output = DEST / 'MIA_5query_adaptive.html'
    output.write_text(html, encoding='utf-8')
    report = {'html_bytes': output.stat().st_size,
              'html_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'format': 'single offline HTML with both complete per-team policies',
              'single_team_states': [11, 55], 'single_team_max_depth': [4, 5],
              'guarantee_method': policy['scope'],
              'source_files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                               'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in [PROOF, encoding_path, padding_path]]}
    artifact_path('size_report_5query_adaptive.json').parent.mkdir(parents=True, exist_ok=True)
    artifact_path('size_report_5query_adaptive.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
