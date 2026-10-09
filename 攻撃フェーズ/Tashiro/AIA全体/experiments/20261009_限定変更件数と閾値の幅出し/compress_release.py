#!/usr/bin/env python3
"""Build deterministic XZ review artifacts from generated sweep CSVs."""
from __future__ import annotations
import argparse,gzip,lzma,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
r=a.results
out={}
for stem,source in [('per_team_grid.csv',r/'per_team_grid.csv.gz'),('grid_summary.csv',r/'grid_summary.csv'),('grid_bootstrap.csv',r/'grid_bootstrap.csv')]:
 data=gzip.decompress(source.read_bytes()) if source.suffix=='.gz' else source.read_bytes()
 target=r/(stem+'.xz')
 target.write_bytes(lzma.compress(data,preset=8))
 assert lzma.decompress(target.read_bytes())==data
 out[target.name]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'source_uncompressed_sha256':hashlib.sha256(data).hexdigest()}
(r/'release_hashes.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('RELEASE_COMPRESSED',list(out.keys()))
