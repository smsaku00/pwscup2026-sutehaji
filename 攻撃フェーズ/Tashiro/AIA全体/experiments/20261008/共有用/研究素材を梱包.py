"""研究素材の共有用一覧を作る。まとめ本文・提出回答は作らない。"""
from pathlib import Path
import hashlib, json, argparse, csv, gzip, base64

ROOT = Path(__file__).resolve().parent.parent
TMP = Path('/private/tmp/pws-aia1008-sources')
REMOTE = '攻撃フェーズ/Tashiro/AIA全体/experiments/20261008/'
EXCLUDED_PARTS = {'作業用', '入力', '__pycache__'}
EXCLUDED_NAMES = {'共有素材一覧.json', 'Pro入力資料.json'}

def git_sha(b):
    return hashlib.sha1(b'blob '+str(len(b)).encode()+bytes([0])+b).hexdigest()

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exclude-folder', action='append', default=[])
    args=parser.parse_args()
    TMP.mkdir(exist_ok=True)
    entries=[]; binary=[]; manifest=[]; local_notes=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in EXCLUDED_PARTS) or p.name in EXCLUDED_NAMES: continue
        if rel.parts[0] in args.exclude_folder: continue
        if p.suffix not in {'.py','.json','.csv','.png','.svg','.md'} and p.name!='.gitignore': continue
        b=p.read_bytes()
        item={'path':str(rel),'remote_path':REMOTE+str(rel), 'bytes':len(b),
              'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha':git_sha(b)}
        if p.suffix=='.md':
            local_notes.append({'path':str(rel), 'content':b.decode('utf-8'), 'sha256':item['sha256']})
            continue
        if p.suffix=='.csv':
            headers=next(csv.reader(b.decode('utf-8').splitlines()))
            assert not {'record_id','challenge_row_id','is_member','time','SBP','TG','HDL','ALT','FPG','BMI'}.intersection(headers), str(rel)
        if p.suffix=='.csv' and len(b)>=131072:
            packed=gzip.compress(b,mtime=0)
            item.update(remote_path=item['remote_path']+'.gz',remote_encoding='gzip',
                        remote_bytes=len(packed),remote_git_blob_sha=git_sha(packed),
                        remote_sha256=hashlib.sha256(packed).hexdigest())
            binary.append({'path':item['remote_path'],'encoding':'base64',
                           'content':base64.b64encode(packed).decode(), 'expected_sha':git_sha(packed)})
            manifest.append(item)
            continue
        manifest.append(item)
        if p.suffix=='.png':
            binary.append({'path':item['remote_path'],'encoding':'base64',
                           'content':base64.b64encode(b).decode(),'expected_sha':git_sha(b)})
            continue
        entries.append({'path':item['remote_path'],'mode':'100644','type':'blob',
                        'content':b.decode('utf-8'),'expected_sha':git_sha(b)})
    payload=json.dumps({'text_entries':entries,'binary_blobs':binary},ensure_ascii=False,separators=(',',':'))
    (TMP/'tree-entries.json').write_text(payload,encoding='utf-8')
    chunks=[payload[i:i+32000] for i in range(0,len(payload),32000)]
    for i,chunk in enumerate(chunks): (TMP/f'chunk-{i:04d}.json').write_text(json.dumps(chunk,ensure_ascii=False),encoding='utf-8')
    manifest_record={'purpose':'research code and aggregate evidence; no final summary prose',
        'remote_prefix':REMOTE,'markdown_publication_reserved_for_ChatGPT_GPT6_Pro':True,
        'excluded_folders':args.exclude_folder,'files':manifest,'payload_characters':len(payload),
        'payload_sha256':hashlib.sha256(payload.encode()).hexdigest(),'chunks':len(chunks)}
    (ROOT/'共有用/共有素材一覧.json').write_text(json.dumps(manifest_record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (TMP/'manifest.json').write_text(json.dumps(manifest_record,ensure_ascii=False)+'\n',encoding='utf-8')
    # 実験ごとの根拠記録をそのまま引き継ぐ。全体まとめの文章はこの処理で作らない。
    timing_path=ROOT/'共有用/まとめ依頼時刻変更.json'
    timing=json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.exists() else {}
    pro={'purpose':'source material for the user-authorized GPT-6 Pro summary',
        'deadline':'2026-10-08T09:00:00+09:00',
        'summary_request_time':timing.get('summary_request_time','2026-10-08T09:00:00+09:00'),
        'timing_authorization':timing,
        'research_notes':local_notes,'source_manifest':manifest_record}
    (ROOT/'共有用/Pro入力資料.json').write_text(json.dumps(pro,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(manifest),'text_files':len(entries),'binary_files':len(binary),'notes':len(local_notes),
        'chunks':len(chunks),'characters':len(payload),'payload_sha256':manifest_record['payload_sha256']},ensure_ascii=False))

if __name__=='__main__':run()
