#!/usr/bin/env node
'use strict';
// Run the generated page's scripts in a simulated DOM. Check logic, SVG
// content, and downloads without claiming browser rendering or visual QA.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const five = require('./five_core.js');
const shared = require('./tree_core.js');
const root = path.resolve(__dirname, '../..');
const htmlPath = path.join(root, 'docs/visualizer/MIA_5query_cards.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 5);
const data = JSON.parse(scripts[2][2]);
const report = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/size_report_5query.json'), 'utf8'));
const source = JSON.parse(fs.readFileSync(path.join(root, 'docs/検証資料/追加5回の全候補確認.json'), 'utf8'));
const clean = node => Object.hasOwn(node, 'member_mask') ? { member_mask: node.member_mask } : {
  query_mask: node.query_mask, before: node.before,
  children: Object.fromEntries(Object.entries(node.children).map(([h, child]) => [h, clean(child)])),
};
assert.deepEqual(data.policies, Object.fromEntries(Object.entries(source.policies).map(([cap, node]) => [cap, clean(node)])));
assert.deepEqual(data.ids, source.candidate_ids);
assert.deepEqual(data.teams, source.team_order);
assert.deepEqual(data.caps, source.caps);
assert.deepEqual(data.high, source.total_high);
assert.deepEqual(data.demo_masks, [64, 1025, 2, 260, 512, 24]);
assert.equal(Buffer.byteLength(html), report.html_bytes);
assert.equal(crypto.createHash('sha256').update(html).digest('hex'), report.html_sha256);
for (const file of report.source_files) {
  const bytes = fs.readFileSync(path.join(root, file.path));
  assert.equal(bytes.length, file.bytes);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), file.sha256);
}
const result = {
  status: 'passed', environment: 'Node VM with simulated DOM; generated page scripts and SVG markup',
  html_bytes: Buffer.byteLength(html), html_sha256: report.html_sha256,
  source_policies_matched: 2, demo_team10_counts: [55], demo_means_exact: [],
  controls_checked: [], negative_cases: [], visual_browser_validation: false,
};

function simulatedViewer(initialHash = '') {
  const ids = new Map(), downloads = [], blobs = new Map(), replaceCalls = [];
  class Element {
    constructor(tag = 'div', id = '') {
      this.tagName = tag.toUpperCase(); this.id = id; this.value = ''; this.checked = false;
      this.hidden = false; this.disabled = false; this.textContent = ''; this.children = []; this.dataset = {}; this._html = '';
    }
    set innerHTML(value) {
      this._html = String(value); this.children = [];
      const options = [...this._html.matchAll(/<option\b([^>]*)\bvalue="([^"]+)"([^>]*)>/g)];
      if (this.tagName === 'SELECT' && options.length) {
        this.value = (options.find(m => (m[1] + m[3]).includes('selected')) || options[0])[2];
      }
      this.selects = [...this._html.matchAll(/<select\b[^>]*\bdata-team="(\d+)"[^>]*>([\s\S]*?)<\/select>/g)].map(m => {
        const el = new Element('select'); el.dataset.team = m[1]; el.innerHTML = m[2]; return el;
      });
    }
    get innerHTML() { return this._html; }
    replaceChildren(...children) { this.children = children; this._html = ''; this.selects = []; }
    append(el) { this.children.push(el); }
    querySelectorAll(selector) { assert.equal(selector, '[data-team]'); return this.selects || []; }
    click() {
      if (this.disabled) return;
      if (this.tagName === 'A') { downloads.push({ name: this.download, blob: blobs.get(this.href) }); return; }
      return this.onclick?.();
    }
  }
  for (const m of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) ids.set(m[2], new Element(m[1], m[2]));
  ids.get('five-data').textContent = scripts[2][2]; ids.get('error').hidden = true;
  const location = { hash: initialHash, pathname: '/MIA_5query_cards.html', search: '' };
  const document = {
    getElementById(id) { const el = ids.get(id); assert(el, `Unknown DOM id ${id}`); return el; },
    createElement(tag) { return new Element(tag); },
  };
  const sandbox = {
    document, location, history: { replaceState(_state, _title, url) { replaceCalls.push(url); const i = url.indexOf('#'); location.hash = i < 0 ? '' : url.slice(i); } },
    Blob, URL: { createObjectURL(blob) { const key = `blob:test-${blobs.size}`; blobs.set(key, blob); return key; }, revokeObjectURL() {} },
    setTimeout(callback) { queueMicrotask(callback); return 0; }, console,
  };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(scripts[0][2], context, { filename: 'embedded-shared-core.js' });
  vm.runInContext(scripts[1][2], context, { filename: 'embedded-five-core.js' });
  vm.runInContext(scripts[3][2], context, { filename: 'embedded-five-viewer.js' });
  vm.runInContext(scripts[4][2], context, { filename: 'embedded-five-lesson.js' });
  const execute = source => vm.runInContext(source, context);
  const snapshot = () => JSON.parse(execute(`JSON.stringify({history:history.map(v=>v.slice()),counts:state.map(s=>s.candidates.length),
    complete:state.map(s=>s.complete),queries:state.map(s=>s.query),candidates:state.map(s=>s.candidates.slice()),
    pending:pending.slice(),focus,chosen,busy})`));
  const click = id => ids.get(id).click();
  function response(t, h) {
    const el = ids.get('responses').querySelectorAll('[data-team]').find(el => Number(el.dataset.team) === t);
    assert(el); assert(new RegExp(`value="${h}"`).test(el.innerHTML), 'Response is not a rendered local option');
    el.value = String(h); el.onchange();
  }
  function focusTeam(t) { ids.get('focus').value = String(t); ids.get('focus').onchange(); }
  function branch(h) {
    const el = ids.get('branches').children.find(el => el.innerHTML.includes(`>${h}人</span>`));
    assert(el); el.click();
  }
  function card(member) {
    const label = shared.maskIDs(member, data.ids).join('・');
    const el = ids.get('cards').children.find(el => el.innerHTML.includes(`<b>${label}</b>`));
    assert(el); el.click();
  }
  return { ids, context, execute, snapshot, click, response, focusTeam, branch, card, downloads, replaceCalls, location };
}

(async function main() {
  const viewer = simulatedViewer();
  assert.deepEqual(viewer.snapshot().counts, [11, 55, 11, 55, 11, 55]);
  assert.equal(viewer.snapshot().focus, 1);
  assert.equal(viewer.ids.get('cards').children.length, 55);
  assert.equal(viewer.ids.get('branches').children.length, 3);
  assert(viewer.ids.get('back').disabled);
  for (const id of data.ids) assert(viewer.ids.get('board').innerHTML.includes(`>${id}</text>`));
  assert.equal((viewer.ids.get('board').innerHTML.match(/r="9" fill="#c76722"/g) || []).length, 2);
  assert.equal((viewer.ids.get('board').innerHTML.match(/stroke="#087d83"/g) || []).length, 4);
  assert(viewer.ids.get('board-note').textContent.includes('47・675'));
  const oldHistory = viewer.snapshot().history;
  viewer.branch(0);
  assert.equal(viewer.snapshot().pending[1], 0);
  assert.deepEqual(viewer.snapshot().history, oldHistory);
  assert(viewer.ids.get('cards-note').textContent.includes('残る数は21'));
  viewer.branch(1); viewer.card(3);
  assert.equal(viewer.snapshot().chosen, 3);
  assert.deepEqual(viewer.snapshot().history, []);
  assert(viewer.ids.get('intersection').textContent.includes('予想回答2人'));
  assert(viewer.ids.get('intersection').textContent.includes('一致しない'));
  viewer.card(1025);
  result.controls_checked.push('actual IDs in SVG', 'pair-card intersection', 'response preview without navigation', 'candidate-card inspection');

  const means = ['233/18000', '169/9000', '7/2000', '553/27000', '11/6000'];
  const counts = [[6, 28, 5, 28, 6, 28], [3, 13, 2, 12, 3, 13], [2, 5, 1, 6, 2, 4], [1, 2, 1, 2, 1, 1], [1, 1, 1, 1, 1, 1]];
  for (let step = 0; step < 5; step++) {
    assert(viewer.ids.get('pending-mean').textContent.includes(means[step]));
    result.demo_means_exact.push(means[step]);
    const before = viewer.snapshot();
    before.queries.forEach((q, t) => {
      const padding = data.high[t] - shared.popcount(q);
      assert(padding >= 1 && padding <= 13);
      assert(viewer.ids.get('padding').innerHTML.includes(`<td>${data.teams[t]}</td><td>${shared.popcount(q)}</td><td title="${data.negative_ids.slice(0, padding).join('、')}">${padding}</td><td>${data.high[t]}</td>`));
    });
    viewer.click('demo');
    assert(viewer.ids.get('error').hidden);
    assert.deepEqual(viewer.snapshot().counts, counts[step]);
    result.demo_team10_counts.push(viewer.snapshot().counts[1]);
    assert.equal(viewer.ids.get('stack').children.length, step + 1);
    if (step === 2) {
      viewer.focusTeam(2);
      assert(viewer.snapshot().complete[2]); assert.equal(viewer.snapshot().queries[2], 0);
      assert.equal(viewer.snapshot().pending[2], 0);
      assert.equal(viewer.ids.get('branches').children.length, 1);
      assert.equal((viewer.ids.get('board').innerHTML.match(/r="9" fill="#c76722"/g) || []).length, 1);
      assert.equal((viewer.ids.get('board').innerHTML.match(/stroke="#087d83"/g) || []).length, 0);
      viewer.focusTeam(1);
    }
  }
  assert.deepEqual(result.demo_team10_counts, [55, 28, 13, 5, 2, 1]);
  assert(viewer.snapshot().complete.every(Boolean));
  assert.deepEqual(viewer.snapshot().queries, [0, 0, 0, 0, 0, 0]);
  assert(viewer.ids.get('next').disabled);
  assert(viewer.ids.get('complete').innerHTML.includes('Team 10：47・675'));
  assert(viewer.ids.get('complete').innerHTML.includes('対象6 Teamに所属しない候補：264・383'));
  viewer.click('export');
  const download = viewer.downloads.at(-1);
  assert.equal(download.name, 'MIA_5query_path.json');
  const exported = JSON.parse(await download.blob.text());
  assert(exported.complete);
  assert.deepEqual(exported.remaining_counts, [1, 1, 1, 1, 1, 1]);
  assert.deepEqual(exported.members_by_team['10'], [47, 675]);
  assert.deepEqual(exported.members_by_team['6'], [304]);
  result.controls_checked.push('full five-question demo', 'one-member Team early completion and zero placeholders', 'padding rows', 'JSON path export');

  viewer.card(3);
  assert(viewer.ids.get('board-note').textContent.includes('すでに除外されています'));
  assert(viewer.ids.get('stack').children.some(el => el.className.includes('bad')));
  assert.deepEqual(viewer.snapshot().counts, [1, 1, 1, 1, 1, 1]);
  viewer.ids.get('remaining-only').checked = true; viewer.ids.get('remaining-only').onchange();
  assert.equal(viewer.ids.get('cards').children.length, 1);
  viewer.ids.get('remaining-only').checked = false; viewer.ids.get('remaining-only').onchange();
  assert.equal(viewer.ids.get('cards').children.length, 55);
  viewer.click('back'); assert.equal(viewer.snapshot().history.length, 4);
  assert.deepEqual(viewer.snapshot().counts, counts[3]);
  viewer.ids.get('steps').children[2].click(); assert.equal(viewer.snapshot().history.length, 2);
  assert.deepEqual(viewer.snapshot().counts, counts[1]);
  viewer.click('reset'); assert.deepEqual(viewer.snapshot().history, []);
  assert.deepEqual(viewer.snapshot().counts, [11, 55, 11, 55, 11, 55]);
  result.controls_checked.push('removed-card stack mismatches without state mutation', 'remaining-only card filter', 'back and earlier step', 'reset');

  // A locally legal answer in every Team may violate exclusive assignment.
  for (let t = 0; t < 6; t++) viewer.response(t, 0);
  const beforeInvalid = viewer.snapshot();
  viewer.click('next'); assert(!viewer.ids.get('error').hidden);
  assert(viewer.ids.get('error').textContent.includes('複数Team'));
  assert.deepEqual(viewer.snapshot(), beforeInvalid);
  result.negative_cases.push('locally legal all-zero vector has no joint placement and does not navigate');
  viewer.ids.get('mean').value = '0'; viewer.click('decode');
  assert(!viewer.ids.get('error').hidden); assert.deepEqual(viewer.snapshot().history, []);
  result.negative_cases.push('capacity-legal decoded mean with no joint placement');
  const scoreValues = Object.values(data.encoding.integer_values).sort((a, b) => a - b);
  viewer.ids.get('mean').value = `${scoreValues[0] + scoreValues[1]}/${data.encoding.mean_denominator * 2}`;
  viewer.click('decode'); assert(!viewer.ids.get('error').hidden); assert.deepEqual(viewer.snapshot().history, []);
  viewer.ids.get('mean').value = '1'; viewer.click('decode'); assert(!viewer.ids.get('error').hidden);
  assert.deepEqual(viewer.snapshot().history, []);
  result.negative_cases.push('midpoint mean', 'out-of-margin mean');
  viewer.click('reset'); viewer.ids.get('mean').value = means[0]; viewer.click('decode');
  assert(viewer.ids.get('error').hidden); assert.deepEqual(viewer.snapshot().counts, counts[0]);
  viewer.click('demo'); viewer.click('demo');
  assert.equal(viewer.snapshot().history.length, 3);
  const validPending = viewer.snapshot().pending; validPending[2] = 1;
  const code = validPending.reduce((sum, h, t) => sum + h * 3 ** t, 0);
  viewer.ids.get('mean').value = String(data.encoding.integer_values[code] / data.encoding.mean_denominator);
  const beforeFinished = viewer.snapshot(); viewer.click('decode');
  assert(!viewer.ids.get('error').hidden); assert.deepEqual(viewer.snapshot(), beforeFinished);
  result.negative_cases.push('nonzero decoded hit for completed Team leaves route unchanged');
  result.controls_checked.push('reachable mean navigation');

  viewer.click('example-end'); assert(viewer.snapshot().complete.every(Boolean));
  const hash = viewer.location.hash;
  assert(hash.startsWith('#h=')); assert(viewer.replaceCalls.length > 0);
  const restored = simulatedViewer(hash);
  assert.deepEqual(restored.snapshot().history, viewer.snapshot().history);
  assert.deepEqual(restored.snapshot().counts, [1, 1, 1, 1, 1, 1]);
  result.controls_checked.push('full-example shortcut', 'deep route restoration and URL replacement');
  if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_five_ui.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
