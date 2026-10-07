"""Package the organized offline documents, preserving their relative links."""
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from artifact_layout import artifact_path

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'docs' / 'visualizer'
REPO = HERE.parents[4]
FILES = {str(p.relative_to(DEST)): p for p in DEST.rglob('*') if p.is_file() and p.suffix != '.zip'}
FILES.update({'resources/verification/' + p.name: p for p in HERE.glob('validation_*.json')})


def portable_markdown(path, content):
    """Keep packaged references local; link remaining repository sources to GitHub."""
    packaged = {source.resolve(): name for name, source in FILES.items()}

    def replace(match):
        label, href = match.group(1), match.group(2)
        if href.startswith(('https:', 'http:', '#', 'mailto:')):
            return match.group(0)
        target, separator, fragment = href.partition('#')
        absolute = (path.parent / unquote(target)).resolve()
        if absolute in packaged:
            relative = os.path.relpath(packaged[absolute], Path(packaged[path.resolve()]).parent).replace(os.sep, '/')
        elif absolute.is_dir() and absolute.is_relative_to(DEST.resolve()):
            return match.group(0)
        else:
            kind = 'tree' if absolute.is_dir() else 'blob'
            relative = 'https://github.com/smsaku00/pwscup2026-sutehaji/' + kind + '/main/' + quote(str(absolute.relative_to(REPO.resolve())), safe='/')
        return '[' + label + '](' + relative + (separator + fragment if separator else '') + ')'

    return re.sub(r'\[([^\]\n]+)\]\(([^)\s]+)\)', replace, content)


def main():
    output = artifact_path('MIA_4query_visualizer_bundle.zip')
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = {name: portable_markdown(path, path.read_text(encoding='utf-8')).encode('utf-8') if path.suffix == '.md' else path.read_bytes()
               for name, path in sorted(FILES.items())}
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in entries.items():
            entry = ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, content, compress_type=ZIP_DEFLATED, compresslevel=9)
    with ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('damaged archive')
        for name, content in entries.items():
            if bundle.read(name) != content:
                raise ValueError('archive content mismatch: ' + name)
    print(str(output), output.stat().st_size, 'bytes;', len(entries), 'files')


if __name__ == '__main__':
    main()
