#!/usr/bin/env node
'use strict';
// Execute the generated HTML's actual scripts in a simulated DOM. This checks
// integration and downloads; it does not claim browser rendering or visual QA.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const core = require('./tree_core.js');
const root = path.resolve(__dirname, '../..');
const htmlPath = path.join(root, 'docs/visualizer/MIA_4query_tree.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 3);
const embedded = JSON.parse(scripts[1][2]);
const sizes = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/size_report.json')));
const proof = path.join(root, 'analysis/full_depth4_20261007');
const sourceHashes = new Map(sizes.source_files.map(f => [f.path, f.sha256]));
const result = {
  status: 'passed', environment: 'Node VM with simulated DOM; native GZIP DecompressionStream',
  html_bytes: Buffer.byteLength(html), html_sha256: crypto.createHash('sha256').update(html).digest('hex'),
  embedded_certificates_source_matched: 0, native_gzip_certificates_loaded: 0,
  demo_remaining_counts: [], transported_branch_history: [],
  controls_checked: [], negative_cases: [], visual_browser_validation: false,
};
assert.equal(result.html_bytes, sizes.html_bytes);
for (const orbit of embedded.manifest.orbits) {
  const rep = orbit.representative, relative = `orbits/${rep}/certificate.json.gz`;
  const gzipBytes = Buffer.from(embedded.certificates[rep], 'base64');
  assert.equal(crypto.createHash('sha256').update(gzipBytes).digest('hex'), sourceHashes.get(relative));
  assert(gzipBytes.equals(fs.readFileSync(path.join(proof, relative))));
  const cert = JSON.parse(zlib.gunzipSync(gzipBytes));
  assert.equal(cert.q3_code, rep);
  assert.equal(cert.depth, 3);
  result.embedded_certificates_source_matched++;
}
assert.equal(result.embedded_certificates_source_matched, 53);

function simulatedViewer(initialHash = '') {
  const ids = new Map(), downloads = [], objectUrls = new Map(), replaceCalls = [];
  class Element {
    constructor(tag = 'div', id = '') {
      this.tagName = tag.toUpperCase(); this.id = id; this.value = ''; this.hidden = false;
      this.disabled = false; this.textContent = ''; this.children = []; this.dataset = {}; this._html = '';
    }
    set innerHTML(value) {
      this._html = String(value); this.children = [];
      this.answerElements = [...this._html.matchAll(/<tr\b[^>]*data-code="(\d+)"[^>]*>/g)].map(match => {
        const el = new Element('tr'); el.dataset.code = match[1]; return el;
      });
    }
    get innerHTML() { return this._html; }
    replaceChildren(...children) { this.children = children; this._html = ''; this.answerElements = []; }
    append(el) { this.children.push(el); }
    querySelectorAll(selector) {
      if (selector === '[data-code]') return this.answerElements || [];
      throw new Error(`Unexpected simulated element selector: ${selector}`);
    }
    click() {
      if (this.disabled) return;
      if (this.tagName === 'A') { downloads.push({ name: this.download, blob: objectUrls.get(this.href) }); return; }
      return this.onclick?.();
    }
  }
  for (const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) ids.set(match[2], new Element(match[1], match[2]));
  const buttonIDs = [...html.matchAll(/<button\b[^>]*\bid="([^"]+)"[^>]*>/gi)].map(m => m[1]);
  ids.get('mia-data').textContent = scripts[1][2];
  ids.get('sort').value = 'count'; ids.get('error').hidden = true;
  const location = { hash: initialHash, pathname: '/MIA_4query_tree.html', search: '' };
  const document = {
    getElementById(id) { const el = ids.get(id); assert(el, `Unknown DOM id ${id}`); return el; },
    createElement(tag) { return new Element(tag); },
    querySelectorAll(selector) {
      assert.equal(selector, 'button');
      return [...buttonIDs.map(id => ids.get(id)), ...ids.get('steps').children.filter(el => el.tagName === 'BUTTON')];
    },
  };
  const browserHistory = { replaceState(_state, _title, url) {
    replaceCalls.push(url); const index = url.indexOf('#'); location.hash = index < 0 ? '' : url.slice(index);
  } };
  const sandbox = {
    document, location, history: browserHistory, Blob, Response, DecompressionStream, atob,
    URL: { createObjectURL(blob) { const key = `blob:test-${objectUrls.size}`; objectUrls.set(key, blob); return key; }, revokeObjectURL() {} },
    setTimeout(callback) { queueMicrotask(callback); return 0; }, console,
  };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(scripts[0][2], context, { filename: 'embedded-tree-core.js' });
  vm.runInContext(scripts[2][2], context, { filename: 'embedded-viewer.js' });
  const execute = source => vm.runInContext(source, context);
  const snapshot = () => JSON.parse(execute(`JSON.stringify({history:[...history],remaining:rows?rows.length:data.summary.total_states,
    complete:!!node.leaf,terminal:!!node.terminal,query:node.query||null,outcomes:[...outcomes].map(([code,g])=>[code,g.length]),
    routeCounts:[...routeCounts],page,busy,certCacheSize:certCache.size})`));
  async function settled() {
    // No browser or network: wait for native in-process stream promises.
    for (let i = 0; i < 10000 && snapshot().busy; i++) await new Promise(resolve => setImmediate(resolve));
    assert(!snapshot().busy, 'Viewer action did not finish');
  }
  async function click(id) { await ids.get(id).click(); await settled(); }
  async function answer(code) {
    const el = ids.get('answers').querySelectorAll('[data-code]').find(e => Number(e.dataset.code) === code);
    assert(el, `Answer ${code} is not rendered`); await el.click(); await settled();
  }
  return { ids, context, execute, snapshot, settled, click, answer, downloads, replaceCalls, location };
}

(async function main() {
  const viewer = simulatedViewer(); await viewer.settled();
  assert.equal(viewer.snapshot().remaining, 2494800);
  assert.equal(viewer.snapshot().outcomes.length, 209);
  assert(viewer.ids.get('back').disabled);
  assert(viewer.ids.get('csv').disabled);
  assert(viewer.ids.get('prev-page').disabled);
  assert(viewer.ids.get('next-page').disabled);
  // Every inline GZIP is inflated through the generated viewer's own loader.
  for (const orbit of embedded.manifest.orbits) {
    const cert = await viewer.execute(`certificate(${orbit.representative})`);
    assert.equal(cert.q3_code, orbit.representative);
    assert.equal(cert.depth, 3);
    assert.equal(cert.state_hash, JSON.parse(zlib.gunzipSync(Buffer.from(embedded.certificates[orbit.representative], 'base64'))).state_hash);
    assert(viewer.snapshot().certCacheSize <= 3);
    result.native_gzip_certificates_loaded++;
  }
  for (const expected of [40428, 631, 20, 1]) {
    await viewer.click('demo'); assert.equal(viewer.snapshot().remaining, expected);
    assert(viewer.ids.get('error').hidden); result.demo_remaining_counts.push(expected);
  }
  assert.deepEqual(viewer.snapshot().history, [30, 253, 246, 328]);
  assert(viewer.snapshot().complete);
  assert(viewer.ids.get('question').innerHTML.includes('どこにも属さない候補'));
  assert.equal(viewer.snapshot().outcomes.length, 0);
  assert(viewer.ids.get('demo').disabled);
  await viewer.click('export');
  const pathDownload = viewer.downloads.at(-1);
  assert.equal(pathDownload.name, 'MIA_4query_path.json');
  const exported = JSON.parse(await pathDownload.blob.text());
  assert(exported.complete);
  assert.deepEqual(exported.codes, [30, 253, 246, 328]);
  assert.deepEqual(exported.members_by_team['6'], [264]);
  assert.deepEqual(exported.members_by_team['10'], [90, 115]);
  result.controls_checked.push('four-step demo', 'leaf assignment with two unassigned candidates', 'JSON path export');

  await viewer.click('back');
  assert.deepEqual(viewer.snapshot().history, [30, 253, 246]);
  assert.equal(viewer.snapshot().remaining, 20);
  assert(viewer.snapshot().terminal);
  assert(viewer.ids.get('prev-page').disabled);
  assert(viewer.ids.get('next-page').disabled);
  await viewer.click('csv');
  const csvDownload = viewer.downloads.at(-1);
  assert.equal(csvDownload.name, 'MIA_remaining_candidates.csv');
  const csv = await csvDownload.blob.text();
  assert.equal(csv.split('\r\n').length, 21);
  assert(csv.includes('Team6,Team10,Team11,Team16,Team17,Team25,response_code,response_counts'));
  assert.equal(new Set(csv.split('\r\n').slice(1).map(line => line.split(',')[7])).size, 20);
  assert(viewer.ids.get('question').innerHTML.includes('Team 6'));
  assert(viewer.ids.get('question').innerHTML.includes('264'));
  assert(viewer.ids.get('question').innerHTML.includes('調べる候補ID'));
  assert.equal(viewer.ids.get('team-order').textContent, 'Team 6・Team 10・Team 11・Team 16・Team 17・Team 25');
  assert(viewer.ids.get('mapping').textContent.includes('Team 10＝2人'));
  result.controls_checked.push('back', 'terminal candidate CSV export', 'actual candidate IDs and Team labels');

  await viewer.click('reset'); assert.deepEqual(viewer.snapshot().history, []);
  viewer.ids.get('filter').value = '253'; viewer.ids.get('filter').oninput();
  assert.equal(viewer.ids.get('answers').querySelectorAll('[data-code]').length, 1);
  viewer.ids.get('filter').value = ''; viewer.ids.get('filter').oninput();
  viewer.ids.get('sort').value = 'code'; viewer.ids.get('sort').onchange();
  const renderedCodes = viewer.ids.get('answers').querySelectorAll('[data-code]').map(e => Number(e.dataset.code));
  assert.deepEqual(renderedCodes, [...renderedCodes].sort((a, b) => a - b));
  viewer.ids.get('mean').value = String(embedded.encoding.integer_values[10] / embedded.encoding.mean_denominator);
  await viewer.click('decode');
  assert(!viewer.ids.get('error').hidden);
  assert(viewer.ids.get('error').textContent.includes('現在の経路には存在しません'));
  assert.deepEqual(viewer.snapshot().history, []);
  const sortedScores = Object.values(embedded.encoding.integer_values).sort((a, b) => a - b);
  viewer.ids.get('mean').value = `${sortedScores[0] + sortedScores[1]}/${embedded.encoding.mean_denominator * 2}`;
  await viewer.click('decode'); assert(!viewer.ids.get('error').hidden);
  assert.deepEqual(viewer.snapshot().history, []);
  viewer.ids.get('mean').value = '1'; await viewer.click('decode'); assert(!viewer.ids.get('error').hidden);
  assert.deepEqual(viewer.snapshot().history, []);
  result.controls_checked.push('reset', 'answer filtering and sorting');
  result.negative_cases.push('legal but unreachable decoded answer', 'midpoint score', 'out-of-margin score');

  // Choose an alternative transported q3 branch through its rendered row,
  // then compare each displayed question with the source-derived core tree.
  const altCode = 246;
  const orbit = embedded.manifest.orbits.find(o => o.q3_codes.includes(altCode));
  assert.notEqual(orbit.representative, altCode);
  const mapping = orbit.mappings[altCode];
  const cert = JSON.parse(zlib.gunzipSync(Buffer.from(embedded.certificates[orbit.representative], 'base64')));
  let expectedNode = core.normalize(core.moveTree(cert.tree, mapping.candidate_old_to_new, mapping.team_old_to_new));
  let expectedRows = core.generate(altCode);
  await viewer.answer(altCode);
  assert.equal(viewer.snapshot().remaining, expectedRows.length);
  assert.deepEqual(viewer.snapshot().query, expectedNode.query);
  assert(!viewer.ids.get('next-page').disabled);
  await viewer.click('next-page'); assert.equal(viewer.snapshot().page, 1);
  assert(!viewer.ids.get('prev-page').disabled);
  await viewer.click('prev-page'); assert.equal(viewer.snapshot().page, 0);
  assert(viewer.ids.get('prev-page').disabled);
  for (let round = 4; round <= 6; round++) {
    assert.deepEqual(viewer.snapshot().query, expectedNode.query);
    const groups = core.groups(expectedRows, expectedNode.query);
    const selectedCode = [...groups].sort((a, b) => b[1].length - a[1].length || a[0] - b[0])[0][0];
    expectedRows = groups.get(selectedCode);
    expectedNode = expectedNode.terminal ? { leaf: true } : expectedNode.children[selectedCode];
    if (round === 5) {
      viewer.ids.get('mean').value = String(embedded.encoding.integer_values[selectedCode] / embedded.encoding.mean_denominator);
      await viewer.click('decode'); assert(viewer.ids.get('error').hidden);
    } else await viewer.answer(selectedCode);
    assert.equal(viewer.snapshot().remaining, expectedRows.length);
    assert.equal(viewer.snapshot().complete, !!expectedNode.leaf);
  }
  assert.equal(viewer.snapshot().remaining, 1);
  result.transported_branch_history = viewer.snapshot().history;
  result.controls_checked.push('alternative transported branch through round 6', 'pagination', 'reachable score navigation');
  await viewer.click('reset'); await viewer.click('last20');
  assert.deepEqual(viewer.snapshot().history, [30, 253, 246]);
  assert.equal(viewer.snapshot().outcomes.length, 20);
  assert.equal(viewer.location.hash, '#codes=30,253,246');
  assert(viewer.replaceCalls.length > 0);
  result.controls_checked.push('20-state shortcut', 'shareable route replacement');

  const deep = simulatedViewer('#codes=30,253,246'); await deep.settled();
  assert.deepEqual(deep.snapshot().history, [30, 253, 246]);
  assert.equal(deep.snapshot().remaining, 20);
  result.controls_checked.push('saved hash route restoration');
  const missing = simulatedViewer(); await missing.settled();
  missing.execute('delete data.certificates["30"]'); await missing.answer(30);
  assert(!missing.ids.get('error').hidden);
  assert(missing.ids.get('error').textContent.includes('内蔵証明書がありません'));
  assert.deepEqual(missing.snapshot().history, []);
  assert.equal(missing.snapshot().outcomes.length, 209);
  result.negative_cases.push('missing certificate stops without changing the route');
  const unsupported = simulatedViewer(); await unsupported.settled();
  unsupported.execute('DecompressionStream=undefined'); await unsupported.answer(30);
  assert(!unsupported.ids.get('error').hidden);
  assert.deepEqual(unsupported.snapshot().history, []);
  result.negative_cases.push('unsupported decompression stops without changing the route');
  const concurrent = simulatedViewer(); await concurrent.settled();
  const firstClick = concurrent.answer(283);
  assert(concurrent.snapshot().busy);
  await concurrent.ids.get('reset').onclick(); // Direct invocation tests the guard, even if a control bypasses disabled.
  await firstClick;
  assert.deepEqual(concurrent.snapshot().history, [283]);
  assert.equal(concurrent.snapshot().remaining, 53796);
  result.controls_checked.push('busy navigation guard');
  if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_ui.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
