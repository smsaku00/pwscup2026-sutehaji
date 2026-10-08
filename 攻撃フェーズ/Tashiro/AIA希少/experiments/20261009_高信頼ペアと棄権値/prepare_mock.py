#!/usr/bin/env python3
"""Extract only the audited mock inputs from the user's local archived ZIP.

The archive is NOT downloaded. No token file is extracted or displayed.
A completed output directory is never overwritten. Every extracted item must
match input_sources.json. The outputs contain private input data: keep them local.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def extract(archive: Path, out: Path, manifest_path: Path) -> None:
    if out.exists():raise FileExistsError('Use a new local input directory')
    specs=json.loads(manifest_path.read_text(encoding='utf-8'))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha!=specs['archive_sha256']:
        raise ValueError('Archive differs from the audited snapshot; do not silently substitute input data')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.pws_pair_inputs_',dir=out.parent))
    try:
        with zipfile.ZipFile(archive) as outer:
            for dest,src in specs['files'].items():
                rel=PurePosixPath(dest)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe destination')
                source=src['archive_path']
                if '!' in source:
                    outer_name,inner_name=source.split('!',1)
                    if inner_name!='C.csv':raise ValueError('Unexpected nested input')
                    with zipfile.ZipFile(io.BytesIO(outer.read(outer_name))) as inner:
                        data=inner.read(inner_name)
                else:
                    data=outer.read(source)
                if hashlib.sha256(data).hexdigest()!=src['sha256']:
                    raise ValueError('Input hash mismatch: '+dest)
                p=stage/Path(*rel.parts);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (stage/'input_manifest.json').write_text(json.dumps(specs['files'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        stage.rename(out)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True);raise
    print('Verified local inputs:',out)
    print('Keep this directory private; it is not part of the GitHub result package.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('input_sources.json'))
    a=p.parse_args();extract(a.archive.resolve(),a.out.resolve(),a.manifest.resolve())
