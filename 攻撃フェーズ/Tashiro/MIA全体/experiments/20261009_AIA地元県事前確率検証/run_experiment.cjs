#!/usr/bin/env node
'use strict';

// Official preliminary data only. Dependencies: Node.js 18+ (no packages).
// Recompute prediction scores BEFORE opening released membership truth.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert');

const root = path.resolve(process.argv[2] || '.');
const out = path.resolve(process.argv[3] || './mia_aia_locality_reproduction');
if (fs.existsSync(out) && fs.readdirSync(out).length) {
  throw new Error('Nonempty output directory: ' + out);
}
const prem = path.join(root, '公式データ', '予備戦');
const attack = path.join(prem, 'PWSCup2026_prelim_attack');
const common = path.join(prem, 'PWSCup2026_prelim_common');
const truthDir = path.join(prem, '結果', '正解データ');
const teamDirName = fs.readdirSync(prem).find(x =>
  x.startsWith('PWSCup2026_prelim_team05_'));
assert(teamDirName, 'Team05 directory not found');
const teamDir = path.join(prem, teamDirName);
const csv = filename => {
  const text = fs.readFileSync(filename, 'utf8').replace(/\r/g, '').trim();
  const [first, ...lines] = text.split('\n');
  const keys = first.split(',');
  // Official PWS CSV values contain no embedded commas or quoted newlines.
  return lines.map(line => {
    const values = line.split(',');
    assert.equal(values.length, keys.length, filename);
    return Object.fromEntries(keys.map((key, i) => [key, values[i]]));
  });
};
const own = csv(path.join(teamDir, 'B_self.csv'));
const bg = csv(path.join(common, 'A_bg.csv'));
const ownAIA = csv(path.join(attack, 'aia_challenge_5.csv'));
const meta = JSON.parse(fs.readFileSync(path.join(attack, 'mia_columns.json'), 'utf8'));
const targets = meta.target_columns.map(Number).filter(x => x !== 5).sort((a,b) => a-b);
assert.equal(targets.length, 20);

const qi = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture'];
const key = row => qi.map(k => row[k]).join('|');
const ownAIASignatures = new Set(ownAIA.map(key));
const known = own.map(row => ownAIASignatures.has(key(row)));
const normal = own.flatMap((row,i) => (!known[i] && row.is_rare !== '1') ? [i] : []);
const rareUnknown = own.filter((row,i) => !known[i] && row.is_rare === '1').length;
assert.equal(own.length, 1049);
assert.equal(known.filter(Boolean).length, 163);
assert.equal(known.filter((v,i) => v && own[i].is_rare === '1').length, 13);
assert.equal(rareUnknown, 11);
assert.equal(normal.length, 875);
const ownHomes = new Set(ownAIA.map(row => row.prefecture));
assert.equal(ownHomes.size, 1);
const home = [...ownHomes][0];

const cols = ['age', 'SBP', 'TG', 'HDL', 'ALT'];
const num = (row, col) => {
  const value = Number(row[col]);
  assert(Number.isFinite(value), 'Nonfinite data in ' + col);
  return col === 'TG' || col === 'ALT' ? Math.log1p(Math.max(0,value)) : value;
};
const scales = cols.map(col => {
  const values = bg.map(row => num(row,col)).sort((a,b) => a-b);
  const q25 = values[Math.floor(0.25*(values.length-1))];
  const q75 = values[Math.floor(0.75*(values.length-1))];
  return Math.max(1e-7, (q75-q25)/1.349);
});
const vector = row => cols.map((col,i) => num(row,col)/scales[i]);
const ownVectors = own.map(vector);

const rank = values => {
  const order = values.map((_,i) => i).sort((a,b) => (values[b]-values[a]) || (a-b));
  const ans = new Array(values.length);
  for (let left=0; left<order.length;) {
    let right=left+1;
    while (right<order.length && values[order[right]] === values[order[left]]) right++;
    const val = (values.length-(left+right-1)/2)/values.length;
    for (let k=left; k<right; k++) ans[order[k]] = val;
    left = right;
  }
  return ans;
};
const tprAtFpr = (scores,labels,level=0.01) => {
  const positives = labels.reduce((a,b) => a+b,0);
  const negatives = labels.length-positives;
  const sorted = scores.map((s,i) => ({s,y:labels[i]})).sort((a,b) => b.s-a.s);
  let fp=0, tp=0;
  for (let i=0; i<sorted.length;) {
    let j=i, dp=0, dn=0;
    while (j<sorted.length && sorted[j].s === sorted[i].s) {
      if (sorted[j].y) dp++; else dn++;
      j++;
    }
    if ((fp+dn)/negatives >= level) {
      const frac = dn ? Math.max(0, Math.min(1, (level*negatives-fp)/dn)) : 1;
      return (tp+frac*dp)/positives;
    }
    fp += dn;
    tp += dp;
    i=j;
  }
  return 1;
};

const modes = [
  ['nn',0,0],
  ['nonhome_plus_005',0.05,0],
  ['nonhome_plus_010',0.10,0],
  ['nonhome_plus_020',0.20,0],
  ['nonhome_plus_010_gate080',0.10,0.80],
];
const predictions = new Map();
const hashes = {};

// PHASE 1: Predictions. Do not read any pool_ids file here.
for (const id of targets) {
  const C = csv(path.join(attack, 'C_' + id + '.csv')).map(vector);
  assert.equal(C.length, 1049);
  const near = normal.map(index => {
    const a = ownVectors[index];
    let best = Infinity;
    for (const b of C) {
      let dist=0;
      for (let k=0;k<5;k++) {
        const delta=a[k]-b[k];
        dist += delta*delta;
      }
      if (dist<best) best=dist;
    }
    return -best;
  });
  const baselineRanks = rank(near);
  const scoreByMode = {};
  for (const [name,weight,gate] of modes) {
    const adjusted = baselineRanks.map((value,t) =>
      value + (own[normal[t]].prefecture !== home && value >= gate ? weight : 0));
    const reranked = rank(adjusted);
    const scores = own.map((row,i) => known[i] ? 0 : row.is_rare === '1' ? 0.4 : 0);
    normal.forEach((index,t) => {scores[index]=0.6+0.3*reranked[t];});
    scoreByMode[name] = scores;
  }
  const buff = Buffer.from(Float64Array.from(scoreByMode.nn).buffer);
  hashes[id] = crypto.createHash('sha256').update(buff).digest('hex');
  predictions.set(id, scoreByMode);
}
// PHASE 2: Released truth is used only for audits and evaluation.
const ownPool = new Map(csv(path.join(truthDir,'pool_ids_5.csv')).map(
  row => [row.record_id,row.pool_id]));
const table = [];
for (const id of targets) {
  const targetPool = new Set(csv(path.join(truthDir,'pool_ids_' + id + '.csv')).map(r=>r.pool_id));
  const truth = own.map(row => targetPool.has(ownPool.get(row.record_id)) ? 1 : 0);
  const positive = truth.reduce((a,b) => a+b,0);
  assert(positive>=25 && positive<=27);
  assert(known.every((v,i) => !v || truth[i]===0));
  const aia = csv(path.join(attack,'aia_challenge_' + id + '.csv'));
  const targetHomes = new Set(aia.map(row=>row.prefecture));
  const row = {
    target:id,positives:positive,target_home:[...targetHomes].join('/'),
    self_home_positives:truth.reduce((a,y,i)=>a+(y&&own[i].prefecture===home?1:0),0)
  };
  for (const [name] of modes) {
    row[name]=tprAtFpr(predictions.get(id)[name],truth);
  }
  table.push(row);
}
const avg = name => table.reduce((sum,row)=>sum+row[name],0)/table.length;
const summary = {
  source:'PWS Cup 2026 official preliminary data',self_team:5,
  self_home:home,total_candidates:own.length,known_private:163,
  normal_unknown:875,rare_unknown:11,targets:targets,
  sum_targets_positives:table.reduce((s,r)=>s+r.positives,0),
  sum_self_home_positives:table.reduce((s,r)=>s+r.self_home_positives,0),
  average:Object.fromEntries(modes.map(([name])=>[name,avg(name)])),
  score_hashes_before_truth:hashes,
  note:'Truth files read only after all prediction score arrays were computed. Exploration on this one preliminary round is not independent validation.'
};
fs.mkdirSync(out,{recursive:true});
const headers = ['target','positives','self_home_positives',...modes.map(x=>x[0])];
const records = [headers.join(',')];
for (const row of table) {
  records.push(headers.map(k => typeof row[k] === 'number' && k!=='target' &&
    k!=='positives' && k!=='self_home_positives' ? row[k].toFixed(10) : row[k]).join(','));
}
fs.writeFileSync(path.join(out,'per_target.csv'),records.join('\n')+'\n');
fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify({summary:summary.average,positives:summary.sum_targets_positives,home_positives:summary.sum_self_home_positives},null,2));
