"""Verify and replay fixed swaps using Python's standard library only.

This replays already selected changes to the exact original Z09; it is not a
general B-to-C anonymizer and does not perform another candidate search.
"""
import argparse,csv,hashlib,io,json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def digest(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(path.read_text())
def decode(raw):return list(csv.reader(io.StringIO(raw.decode('utf-8'),newline='')))
def encode(rows):
    stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\n').writerows(rows)
    return stream.getvalue().encode('utf-8')

def transform(raw,pairs,columns,reverse=False):
    rows=decode(raw)
    assert rows[0]==columns and len(rows)==1050
    for i,j,col in reversed(pairs) if reverse else pairs:
        assert isinstance(i,int) and isinstance(j,int) and 0<=i<1049 and 0<=j<1049 and i!=j
        assert col in ['prefecture','SBP','TG','HDL','ALT']
        k=columns.index(col);rows[i+1][k],rows[j+1][k]=rows[j+1][k],rows[i+1][k]
    return encode(rows)

def verify():
    manifest=load(HERE/'manifest.json')
    for filename,h in manifest['files'].items():
        assert digest((HERE/filename).read_bytes())==h,('file hash mismatch',filename)
    records={r['id']:r for r in load(HERE/'候補情報.json')}
    for cid in manifest['candidates']:
        raw=(HERE/f'C_{cid}.csv').read_bytes();pairs=load(HERE/f'swaps_{cid}.json')
        assert digest(raw)==records[cid]['c_sha256']
        base=transform(raw,pairs,manifest['columns'],reverse=True)
        assert digest(base)==manifest['base']['c_sha256'],('base mismatch',cid)
        assert transform(base,pairs,manifest['columns'])==raw
        rows=decode(raw);original=decode(base)
        assert len(rows)==manifest['rows_per_candidate']+1
        assert all(len(r)==len(manifest['columns']) for r in rows)
        changes=sum(a!=b for r,s in zip(rows[1:],original[1:]) for a,b in zip(r,s))
        assert changes==records[cid]['changed_cells']
        for k in range(len(manifest['columns'])):
            assert sorted(r[k] for r in rows[1:])==sorted(r[k] for r in original[1:])
    print('Verified: 3 C files, hashes, 1,049 rows, 13 columns, common original Z09, swaps and column multisets.')
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-c',type=Path,help='Exact original Z09 CSV; SHA-256 must match manifest.')
    parser.add_argument('--out',type=Path,help='Output directory for replayed candidate CSVs.')
    parser.add_argument('--candidate',choices=['all','labs_n50_r3','prefecture_n50_r1','combined_n10_r3'],default='all')
    args=parser.parse_args()
    if bool(args.base_c)!=bool(args.out):parser.error('--base-c and --out are required together')
    manifest=verify()
    if args.base_c:
        raw=args.base_c.read_bytes()
        if digest(raw)!=manifest['base']['c_sha256']:parser.error('Input is not the exact original Z09 (including row order).')
        args.out.mkdir(parents=True,exist_ok=True)
        ids=manifest['candidates'] if args.candidate=='all' else [args.candidate]
        for cid in ids:
            output=transform(raw,load(HERE/f'swaps_{cid}.json'),manifest['columns'])
            assert output==(HERE/f'C_{cid}.csv').read_bytes()
            dest=args.out/f'C_{cid}.csv'
            if dest.exists() and dest.read_bytes()!=output:raise FileExistsError(dest)
            dest.write_bytes(output);print(dest)

if __name__=='__main__':main()
