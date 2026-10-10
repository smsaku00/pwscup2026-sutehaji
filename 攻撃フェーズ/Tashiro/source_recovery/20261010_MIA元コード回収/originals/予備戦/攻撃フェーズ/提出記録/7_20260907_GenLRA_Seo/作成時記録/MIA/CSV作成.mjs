// Called by 作成.py from a temporary directory with the bundled dependencies.
// The required deliverable is a machine-readable numeric CSV, not an XLSX.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook } from '@oai/artifact-tool';

const here = process.argv[1];
const input = JSON.parse(await fs.readFile(path.join(here,'提出行列.json'),'utf8'));
if (input.header.length !== 21 || input.rows.length !== 1049) throw Error('Shape mismatch');
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('F_mia');
sheet.getRange('A1:U1').values = [input.header];
sheet.getRange('A2:U1050').values = input.rows;
const authored = sheet.getRange('A1:U1050').values;
if (JSON.stringify(authored) !== JSON.stringify([input.header,...input.rows])) throw Error('Numeric authoring mismatch');
console.log((await workbook.inspect({kind:'table',range:'F_mia!A1:E4',include:'values',
  tableMaxRows:4,tableMaxCols:5,maxChars:1600})).ndjson);
console.log((await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
// 一時プレビューだけ表示書式を設定。CSVの値・精度・列構成には影響しない。
sheet.getRange('A1:E4').format.columnWidth=23;
sheet.getRange('A2:A4').setNumberFormat('0');
sheet.getRange('B2:E4').setNumberFormat('0.000000000000');
const preview=await workbook.render({sheetName:'F_mia',range:'A1:E4',scale:1.5,format:'png'});
const previewPath=path.join(process.cwd(),'F_mia_preview.png');
await fs.writeFile(previewPath,new Uint8Array(await preview.arrayBuffer()),{flag:'wx'});
console.log(`一時プレビュー: ${previewPath}`);
for (let i=0;i<authored.length;i++) for (const cell of authored[i]) {
  if (i===0 ? typeof cell!=='string' || /[,"\r\n]/.test(cell) : typeof cell!=='number' || !Number.isFinite(cell))
    throw Error('Invalid CSV cell');
}
// Literal CSV values preserve full float precision; formatting/formulas are not submitted.
const csv = authored.map(row=>row.map(String).join(',')).join('\n')+'\n';
await fs.writeFile(path.join(here,'F_mia.csv'),csv,{flag:'wx'});
console.log('F_mia.csvを新規保存: 1049行、20対象、数値の往復一致。');
