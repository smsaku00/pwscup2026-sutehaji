"""Prepare a separate working copy and path aliases; never run recovered sources."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import unicodedata


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='New, separate runtime root; existing destination refused')
    parser.add_argument('--repository', type=Path, help='Optional checkout containing 公式データ; no dataset copied')
    parser.add_argument('--python', type=Path, default=Path(sys.executable), help='Existing Python 3.12 venv/bin/python')
    args = parser.parse_args()
    bundle, out = args.bundle.resolve(), args.output.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if out.is_relative_to(bundle) or (args.repository and out.is_relative_to(args.repository.resolve())):
        raise ValueError('Runtime must be outside the recovered bundle and repository')
    interpreter = args.python.absolute()
    venv_root = interpreter.parent.parent
    if not interpreter.is_file() or not (venv_root / 'pyvenv.cfg').is_file():
        raise ValueError('Provide an existing Python 3.12 venv/bin/python; this tool does not install dependencies')
    config = (venv_root / 'pyvenv.cfg').read_text(encoding='utf-8')
    version = next((line.split('=', 1)[1].strip() for line in config.splitlines()
                    if line.strip().startswith('version =')), '')
    if not version.startswith('3.12.'):
        raise ValueError('Python 3.12 is required by the project environment instructions')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    source_root = manifest['source_root_original']
    # Validate every source before creating a runtime. Originals remain byte-for-byte untouched.
    contents = []
    for item in manifest['sources']:
        content = (bundle / item['bundle_relative_path']).read_bytes()
        assert digest(content) == item['sha256']
        contents.append((item, content))
    out.mkdir(parents=True)
    transformations, links = [], []
    for item, original in contents:
        target = out / item['original_relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        content = original
        occurrences = original.count(source_root.encode()) if target.suffix == '.py' else 0
        if occurrences:
            content = original.replace(source_root.encode(), str(out).encode())
        target.write_bytes(content)
        transformations.append({'path': item['original_relative_path'], 'original_sha256': item['sha256'],
                                'runtime_sha256': digest(content), 'absolute_root_replacements': occurrences})

    def link(target, source):
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                return
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.absolute(), target_is_directory=source.is_dir())
        links.append({'runtime_relative_path': str(target.relative_to(out)), 'external_or_alias_target': str(source)})

    link(out / '.venv', venv_root)
    archive = out / 'アーカイブ/予備戦_攻撃フェーズ'
    link(out / 'work', archive / 'work_20260901以前')
    link(archive / 'スターターキット', out / 'スターターキット')
    unresolved = ['old saved score/feature NPZ caches and result tables are not included',
                  'legacy share_team05/data, work/data_prelim_team05 and submission history require user-provided data',
                  'starter participant_data and scoring configuration require their exact original distributions']
    if args.repository:
        repo = args.repository.resolve()
        official = repo / '公式データ'
        if not official.is_dir():
            raise FileNotFoundError(official)
        link(out / '予備戦/配布データ', official / '予備戦')
        distributed = out / '本戦/配布データ'
        # Main code uses one team folder and the public attack directory. No token or key is linked.
        own = [p for p in official.iterdir() if p.is_dir() and p.name.startswith('PWSCup2026_final_team28_')]
        # macOS checkout may contain canonically equivalent duplicate directory spellings.
        names = {}
        for p in own:
            names.setdefault(unicodedata.normalize('NFC', p.name), []).append(p)
        if len(names) != 1:
            raise FileNotFoundError('Main team28 data folder is not unique')
        candidates = next(iter(names.values()))
        chosen = max(candidates, key=lambda p: sum((p / name).is_file() for name in ['A_bg.csv', 'B_self.csv']))
        own_dest = distributed / unicodedata.normalize('NFC', chosen.name)
        own_dest.mkdir(parents=True, exist_ok=True)
        for name in ['A_bg.csv', 'B_self.csv', 'B.csv', 'schema.json', 'utility_ref.json', 'scoring_config.yaml']:
            if (chosen / name).is_file():
                link(own_dest / name, chosen / name)
        attacks = official / 'PWSCup2026_final_attack'
        link(distributed / 'PWSCup2026_final_attack', attacks)
        link(out / '本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力', attacks)
    record = {'purpose': 'Preparation only; no recovered source imported or executed',
              'original_root': source_root, 'runtime_root': str(out), 'python_link': str(out / '.venv/bin/python'),
              'transformations': transformations, 'links': links, 'unresolved_runtime_inputs': unresolved,
              'notes': ['Absolute-root replacement occurs only in the separate runtime copies.',
                        'Run only the reviewed entrypoint you intend to reproduce; some old scripts write submission-style files.',
                        'The bundle does not assert all 14864 historical conditions are independently executable without original result inputs.']}
    (out / 'runtime_preparation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'runtime': str(out), 'files': len(contents), 'rewritten_files': sum(x['absolute_root_replacements'] > 0 for x in transformations),
                      'sources_executed': 0, 'unresolved_runtime_inputs': unresolved}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
