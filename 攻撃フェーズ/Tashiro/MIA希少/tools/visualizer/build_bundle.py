"""Package the offline viewer, guide, usage notes and validation reports."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'docs' / 'visualizer'
FILES = {
    'MIA_4query_tree.html': DEST / 'MIA_4query_tree.html',
    'MIA_5query_cards.html': DEST / 'MIA_5query_cards.html',
    'MIA_4query_tree_guide.pdf': DEST / 'MIA_4query_tree_guide.pdf',
    'README.md': DEST / 'README.md',
    'size_report.json': DEST / 'size_report.json',
    'size_report_5query.json': DEST / 'size_report_5query.json',
    'validation_core.json': HERE / 'validation_core.json',
    'validation_ui.json': HERE / 'validation_ui.json',
    'validation_five_core.json': HERE / 'validation_five_core.json',
    'validation_five_ui.json': HERE / 'validation_five_ui.json',
    'validation_lesson_ui.json': HERE / 'validation_lesson_ui.json',
}


def main():
    output = DEST / 'MIA_4query_visualizer_bundle.zip'
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, path in FILES.items():
            entry = ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    with ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('damaged archive')
        for name, path in FILES.items():
            if bundle.read(name) != path.read_bytes():
                raise ValueError('archive content mismatch: ' + name)
    print(str(output), output.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
