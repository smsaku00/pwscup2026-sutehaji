"""Rebuild the 3-page guide from saved certificate inputs.

Requires reportlab and pypdf. Run from any directory:
    python tools/visualizer/build_guide.py
Use --base to select another MIA希少 directory and --font to embed a Japanese TTF.
"""

from pathlib import Path
import argparse
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from pypdf import PdfReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[2],
                    help='MIA希少 directory (defaults to this script’s repository location)')
parser.add_argument('--output',type=Path,help='PDF output path')
parser.add_argument('--font',type=Path,help='Japanese TrueType font to embed')
args = parser.parse_args()
BASE = args.base.resolve()
SRC = BASE / 'analysis/full_depth4_20261007'
OUT = args.output or BASE / 'docs/visualizer/MIA_4query_tree_guide.pdf'
WIT = json.loads((SRC / 'example_witness_code30.json').read_text())
MAN = json.loads((SRC / 'manifest.json').read_text())
TEAMS = [6,10,11,16,17,25]
IDS = MAN['candidate_ids']
CAPS = MAN['caps']
OWNERS = list('ABCDEF')
font_path = args.font or Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
if font_path.is_file():
    FONT = 'EmbeddedJapanese'
    pdfmetrics.registerFont(TTFont(FONT,str(font_path)))
else:
    FONT = 'HeiseiKakuGo-W5'
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    print('Japanese TTF not found; using the standard PDF Japanese CID font. '
          'Use --font to embed a font for distribution.')
PAGE_W, PAGE_H = A4
LEFT, RIGHT = 42, PAGE_W - 42
WIDTH = RIGHT - LEFT
NAVY = colors.HexColor('#14263D')
BLUE = colors.HexColor('#1763A1')
TEAL = colors.HexColor('#087E80')
MUTED = colors.HexColor('#4F6173')
PALE = colors.HexColor('#EFF5FA')
LINE = colors.HexColor('#CCD9E5')
GOLD = colors.HexColor('#FFF4DC')
STYLES = {
    'body': ParagraphStyle('body',fontName=FONT,fontSize=10.3,leading=15.6,textColor=NAVY,wordWrap='CJK'),
    'small': ParagraphStyle('small',fontName=FONT,fontSize=8.5,leading=12.5,textColor=MUTED,wordWrap='CJK'),
    'table': ParagraphStyle('table',fontName=FONT,fontSize=9.5,leading=13.4,textColor=NAVY,wordWrap='CJK'),
    'head': ParagraphStyle('head',fontName=FONT,fontSize=13.2,leading=19,textColor=NAVY,wordWrap='CJK'),
}

def p(text, x, top, width, style='body'):
    obj = Paragraph(text, STYLES[style])
    _, h = obj.wrap(width, PAGE_H)
    obj.drawOn(C, x, top-h)
    return top-h

def text(s, x, y, size=10, color=NAVY):
    C.setFillColor(color); C.setFont(FONT,size); C.drawString(x,y,s)

def box(x, top, width, height, fill=PALE, stroke=LINE, radius=8):
    C.setFillColor(fill); C.setStrokeColor(stroke)
    C.roundRect(x,top-height,width,height,radius,fill=1,stroke=1)

def page_header(page, title, sub):
    C.setFillColor(NAVY); C.rect(0,PAGE_H-11,PAGE_W,11,fill=1,stroke=0)
    text('MIA希少 | 第3〜6回の完全決定木',LEFT,PAGE_H-41,9,MUTED)
    text(title,LEFT,PAGE_H-77,23)
    p(sub,LEFT,PAGE_H-94,WIDTH)
    C.setStrokeColor(LINE); C.line(LEFT,42,RIGHT,42)
    text('証明書: analysis/full_depth4_20261007/  |  2026-10-07',LEFT,27,7.6,MUTED)
    C.setFont(FONT,8); C.drawRightString(RIGHT,27,f'{page} / 3')

def decode(code):
    result = []
    for _ in range(6):
        result.append(code % 3); code //= 3
    return result

def dog_nums(mask):
    return [i+1 for i in range(11) if (mask >> i) & 1]

def members(mask):
    return '・'.join(map(str,dog_nums(mask))) or 'なし'

def tuple_str(code):
    return '(' + ', '.join(map(str,decode(code))) + ')'

def table(rows, x, top, widths, row_height=25, font=9.5):
    tbl=Table(rows,colWidths=widths,rowHeights=[row_height]*len(rows))
    tbl.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),FONT),('FONTSIZE',(0,0),(-1,-1),font),
        ('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('TEXTCOLOR',(0,1),(-1,-1),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),0.7,NAVY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
        ('LINEBELOW',(0,1),(-1,-1),0.35,LINE),
    ]))
    _,h=tbl.wrap(sum(widths),PAGE_H);tbl.drawOn(C,x,top-h)
    return top-h

OUT.parent.mkdir(parents=True,exist_ok=True)
C=canvas.Canvas(str(OUT),pagesize=A4,invariant=1,pageCompression=1)
C.setTitle('MIA希少 4回完全決定木ビューア - 図表ガイド')
C.setAuthor('pwscup2026-sutehaji')
C.setSubject('第3〜第6回の全分岐と犬の具体例')

# Page 1: the complete strategy and the one-route example.
page_header(1,'全分岐から、1本の経路をたどる',
            '第3回の回答に合わせて第4回を選び、同様に第5・6回へ進みます。'
            'このPDFは具体例、HTMLは第3〜6回の全分岐を閲覧できます。')
box(LEFT,680,WIDTH,78,GOLD,colors.HexColor('#ECD79E'))
p('犬の例: 11匹の犬を、飼い主A〜Fに1・2・1・2・1・2匹ずつ割り当てます。'
  '合計9匹が所属し、残る2匹はどの飼い主にも属しません。',LEFT+13,667,WIDTH-26)
p('以下は保存された説明用の仮定経路です。本番の所属結果を示すものではありません。',
  LEFT+13,625,WIDTH-26,'small')

counts=[2494800,40428,631,20,1]
codes=[30,253,246,328]
path_x,path_w,path_top=LEFT,291,579
node_h,gap=46,33
for i,n in enumerate(counts):
    top=path_top-i*(node_h+gap)
    box(path_x,top,path_w,node_h,colors.HexColor('#E8F5F2') if i==4 else PALE)
    text('特定結果' if i==4 else f'第{3+i}回の前',path_x+12,top-18,9.5,MUTED)
    C.setFont(FONT,17);C.setFillColor(TEAL if i==4 else NAVY)
    C.drawRightString(path_x+path_w-12,top-34,f'{n:,}通り')
    if i<4:
        mid=path_x+path_w/2
        C.setStrokeColor(BLUE);C.line(mid,top-node_h-3,mid,top-node_h-gap+5)
        C.setFillColor(BLUE)
        poly=C.beginPath();poly.moveTo(mid-3,top-node_h-gap+9);poly.lineTo(mid,top-node_h-gap+4);poly.lineTo(mid+3,top-node_h-gap+9);poly.close()
        C.drawPath(poly,fill=1,stroke=0)
        box(path_x+47,top-node_h-6,path_w-94,21,colors.white,colors.white,2)
        C.setFont(FONT,9.4);C.setFillColor(BLUE)
        C.drawCentredString(mid,top-node_h-20,f'回答 {tuple_str(codes[i])}')

side_x=LEFT+311;side_w=WIDTH-311
text('図の読み方',side_x,564,12,BLUE)
p('箱は残る所属配置の数。矢印は返ってきた6人の回答です。'
  '回答の順番はA・B・C・D・E・Fです。',side_x,552,side_w)
text('実際には1経路だけ',side_x,466,12,BLUE)
p('第3回は209種類の回答に分かれます。第5回後は全経路で1,340,358分岐ですが、'
  '自分が選んだ枝だけを開いて確認できます。',side_x,453,side_w)
text('全証明書を内蔵',side_x,342,12,BLUE)
p('53個の証明書を圧縮してHTML内に保存。必要な分だけ展開するため、'
  'サーバーや証明書フォルダの選択は不要です。',side_x,329,side_w)

text('HTMLビューアの使い方',LEFT,191,13,BLUE)
p('1. GitHubでHTMLを「Download raw file」から保存し、最新版のChrome・Edge・Safariで開く。'
  '<br/>2. 「具体例を1手進める」で上の経路を確認。「最後の20候補を見る」で3ページ目の表へ。'
  '<br/>3. 回答の行を選ぶと任意の枝へ進めます。「犬の例／実ID・Team」で表示を切り替えます。',
  LEFT,178,WIDTH,'body')
p('平均点の入力は保存済み点数表と現在の到達可能な回答で照合します。'
  '「現在の経路を保存」「残る全候補をCSVで保存」から記録を書き出せます。',LEFT,116,WIDTH,'small')
C.showPage()

# Page 2: four saved question tables, translated faithfully to dog numbers.
page_header(2,'4回の質問: どの犬を見せるか',
            '各飼い主に、表の犬を見せます。飼い主は「この中に自分の犬が何匹いるか」を答えます。'
            '犬の番号はすべて1〜11です。')
cols=2;tw=(WIDTH-20)/2
for qidx in range(4):
    x=LEFT+(tw+20)*(qidx%2);top=675-267*(qidx//2)
    text(f'第{3+qidx}回の質問',x,top,14,BLUE)
    p(f'この例の回答: {tuple_str(codes[qidx])}',x,top-10,tw,'small')
    rows=[['飼い主','見せる犬の番号','回答']]
    for t in range(6):
        rows.append([OWNERS[t],members(WIT['query_masks_normalized'][qidx][t]),str(decode(codes[qidx])[t])])
    table(rows,x,top-37,[38,tw-73,35],row_height=24,font=9.0)
    before,after=counts[qidx],counts[qidx+1]
    p(f'候補: {before:,} → {after:,}通り',x,top-216,tw,'small')

text('犬と実ID、飼い主とTeamの対応',LEFT,145,12,BLUE)
dog_map=' / '.join(f'{i+1}＝{ID}' for i,ID in enumerate(IDS))
p('犬の番号＝実ID: '+dog_map,LEFT,132,WIDTH,'small')
p('A＝Team 6 / B＝Team 10 / C＝Team 11 / D＝Team 16 / E＝Team 17 / F＝Team 25',
  LEFT,102,WIDTH,'small')
p('実提出ではHTMLの実ID表示で追加IDと高側合計を確認してください。'
  'confidenceは選択候補＋追加IDを0.4、その他の希少行を0.1とします（保存済みpadding規則）。',
  LEFT,83,WIDTH,'small')
C.showPage()

# Page 3: all 20 final candidates, including the unassigned dogs.
page_header(3,'最後の20配置は、1回ですべて分かれる',
            '第5回までの回答が (0,1,0,1,0,0) → (1,0,1,0,0,1) → (0,1,0,0,0,1) の場合。'
            '下の数字は犬の番号で、右端は第6回に返るA〜Fの人数です。')
rows=[['配置','A','B','C','D','E','F','所属なし','第6回の回答']]
seen=[]
for num,(state,code) in enumerate(zip(WIT['q5_cell_states'],WIT['q6_response_codes']),1):
    union=0
    for m in state:union |= m
    counts_actual=[(m&q).bit_count() for m,q in zip(state,WIT['query_masks_normalized'][3])]
    assert counts_actual == decode(code)
    seen.append(tuple(counts_actual))
    rows.append([str(num),*[members(m) for m in state],members(((1<<11)-1)^union),
                 '('+','.join(map(str,counts_actual))+')'])
assert len(set(seen)) == len(seen) == 20
bottom=table(rows,LEFT,674,[30,40,55,40,55,40,55,75,WIDTH-390],row_height=22,font=8.8)
box(LEFT,bottom-17,WIDTH,69,colors.HexColor('#E8F5F2'),colors.HexColor('#B3D8D1'))
p('右端の20回答はすべて異なります。第6回の回答と一致する行が1つだけ残るため、'
  'この枝では所属配置を確定できます。例の回答 (1,1,0,0,1,1) は配置1です。',
  LEFT+12,bottom-29,WIDTH-24)
p('照合方法: 各飼い主の所属犬と、2ページ目の第6回で見せる犬との共通数を数えます。'
  '20行の回答は証明書の保存値に加え、所属配置と質問から再計算して一致を確認しました。',
  LEFT,116,WIDTH,'small')
p('出典: manifest.json / example_witness_code30.json / 53個のcertificate.json.gz。'
  '完全な木の検証集計は同じ証明書フォルダのsummary.jsonとindependent_full_audit.jsonを参照。',
  LEFT,83,WIDTH,'small')
C.save()
reader=PdfReader(str(OUT))
assert len(reader.pages)==3
for page in reader.pages:
    assert page.extract_text().strip()
print(json.dumps({'path':str(OUT),'pages':3,'bytes':OUT.stat().st_size,'final_responses_unique':20},ensure_ascii=False))
