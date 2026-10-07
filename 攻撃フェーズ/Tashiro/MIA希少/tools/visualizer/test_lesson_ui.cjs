#!/usr/bin/env node
'use strict';
// Check the beginning lesson with independent enumeration of candidate cards.
// The DOM simulation checks content and controls, not browser rendering.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const htmlPath = path.join(root, 'docs/visualizer/MIA_5query_adaptive.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 5);
const data = JSON.parse(scripts[2][2]);
const source = JSON.parse(fs.readFileSync(path.join(root, 'docs/検証資料/追加5回の全候補確認.json'), 'utf8'));
const pc = mask => mask.toString(2).replace(/0/g, '').length;
const enumerate = cap => Array.from({length: 2047}, (_, i) => i + 1).filter(m => pc(m) === cap);
const independentTrace = (cap, member) => {
  assert.equal(pc(member), cap);
  let candidates = enumerate(cap), node = source.policies[cap];
  const rows = [];
  for (let step = 0; step < 5; step++) {
    const done = Object.hasOwn(node, 'member_mask'), q = done ? 0 : node.query_mask;
    const groups = {};
    for (const candidate of candidates) {
      const h = pc(candidate & q);
      groups[h] = (groups[h] || 0) + 1;
    }
    const h = pc(member & q), before = candidates.slice();
    candidates = candidates.filter(candidate => pc(candidate & q) === h);
    rows.push({step, q, h, before: before.length, after: candidates.length, groups, done,
      before_masks: before, after_masks: candidates.slice()});
    if (!done) node = node.children[h];
  }
  assert.deepEqual(candidates, [member]);
  const signatures = data.ids.map((id, i) => ({id, signature: rows.map(row => String(row.q >> i & 1)).join('')}));
  return {cap, member, rows, signatures};
};

function simulatedViewer() {
  const ids = new Map();
  class Element {
    constructor(tag = 'div', id = '') {
      this.tagName = tag.toUpperCase(); this.id = id; this.value = ''; this.checked = false;
      this.hidden = false; this.disabled = false; this.textContent = ''; this.children = [];
      this.dataset = {}; this.style = {}; this._html = '';
    }
    set innerHTML(value) {
      this._html = String(value); this.children = [];
      const options = [...this._html.matchAll(/<option\b([^>]*)\bvalue="([^"]+)"([^>]*)>/g)];
      if (this.tagName === 'SELECT' && options.length) this.value = (options.find(m => (m[1] + m[3]).includes('selected')) || options[0])[2];
      this.selects = [...this._html.matchAll(/<select\b[^>]*\bdata-team="(\d+)"[^>]*>([\s\S]*?)<\/select>/g)].map(m => {
        const el = new Element('select'); el.dataset.team = m[1]; el.innerHTML = m[2]; return el;
      });
    }
    get innerHTML() { return this._html; }
    replaceChildren(...children) { this.children = children; this._html = ''; this.selects = []; }
    append(...children) { this.children.push(...children); }
    querySelectorAll(selector) { assert.equal(selector, '[data-team]'); return this.selects || []; }
    click() { if (!this.disabled) return this.onclick?.(); }
  }
  for (const m of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) ids.set(m[2], new Element(m[1], m[2]));
  ids.get('five-data').textContent = scripts[2][2]; ids.get('error').hidden = true;
  const location = { hash: '', pathname: '/MIA_5query_cards.html', search: '' };
  const document = {
    getElementById(id) { const el = ids.get(id); assert(el, `Unknown DOM id ${id}`); return el; },
    createElement(tag) { return new Element(tag); },
  };
  const sandbox = { document, location, history: { replaceState(_state, _title, url) {
    const i = url.indexOf('#'); location.hash = i < 0 ? '' : url.slice(i);
  } }, Blob, URL: { createObjectURL() { return 'blob:test'; }, revokeObjectURL() {} },
  setTimeout(callback) { queueMicrotask(callback); return 0; }, console };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  scripts.filter((_, i) => i !== 2).forEach((script, i) => vm.runInContext(script[2], context, {filename: `page-script-${i}.js`}));
  const execute = code => vm.runInContext(code, context);
  const json = code => JSON.parse(execute(`JSON.stringify(${code})`));
  const select = (id, value) => { const el = ids.get(id); assert(el); el.value = String(value); assert(el.onchange); el.onchange(); };
  const click = id => { const el = ids.get(id); assert(el); el.click(); };
  return {ids, execute, json, select, click};
}

const sorted = values => values.slice().sort((a, b) => a - b);
const attrs = markup => Object.fromEntries([...markup.matchAll(/\b([\w-]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));
const colors = ['#2476b8', '#008576', '#bc6913', '#a4497a', '#6552ad'];
function cards(markup) {
  const starts = [...markup.matchAll(/<div class="lesson-member([^\"]*)"([^>]*)>/g)];
  return starts.map((m, i) => ({
    classes: m[1], attrs: attrs(m[2]),
    body: markup.slice(m.index + m[0].length, i + 1 < starts.length ? starts[i + 1].index : markup.length),
  }));
}
function signature(candidate, rows) { return rows.map(row => pc(candidate & row.q) === row.h ? '1' : '0').join(''); }
function assertCards(actual, expected, rows, depth, step = null) {
  assert.equal(actual.length, expected.length);
  assert.deepEqual(sorted(actual.map(card => Number(card.attrs['data-member']))), sorted(expected));
  for (const card of actual) {
    const member = Number(card.attrs['data-member']);
    const sig = signature(member, rows);
    const pass = step === null ? sig.slice(0, depth).split('').every(v => v === '1') : sig[step] === '1';
    assert.equal(card.attrs['data-signature'], sig);
    assert.equal(card.attrs['data-pass'], pass ? '1' : '0');
    assert(card.classes.includes(pass ? 'fits' : 'blocked'));
    const labels = data.ids.filter((_, i) => member >> i & 1).join('<br>');
    assert(card.body.startsWith(`<span>${labels}</span>`));
    if (step === null) {
      const badges = [...card.body.matchAll(/<i class="condition ([^\"]*)"([^>]*)>([1-5])<\/i>/g)];
      assert.equal(badges.length, 5);
      badges.forEach((badge, i) => {
        assert.equal(badge[1], i >= depth ? 'unused' : sig[i] === '1' ? 'agree' : 'reject');
        assert.equal(attrs(badge[2]).style, `--sheet:${colors[i]}`);
        assert.equal(badge[3], String(i + 1));
      });
    }
  }
}

function assertColumn(viewer, cap, member, depth, result) {
  const expected = independentTrace(cap, member), rows = expected.rows;
  const apiRows = viewer.json(`MIA_LESSON.trace(${cap},${member})`);
  assert.equal(apiRows.length, 5);
  const questionMarkup = viewer.ids.get(`lesson-questions-${cap}`).innerHTML;
  const questions = [...questionMarkup.matchAll(/<article\b([^>]*)>([\s\S]*?)<\/article>/g)];
  assert.equal(questions.length, 5);
  questions.forEach((question, i) => {
    const row = rows[i], api = apiRows[i], attributes = attrs(question[1]), body = question[2];
    assert.equal(api.step, i); assert.equal(api.query, row.q); assert.equal(api.hit, row.h);
    assert.equal(api.complete, row.done);
    assert.deepEqual(sorted(api.before), sorted(row.before_masks));
    assert.deepEqual(sorted(api.after), sorted(row.after_masks));
    for (let h = 0; h <= cap; h++) assert.deepEqual(sorted(api.groups[h]), sorted(row.before_masks.filter(m => pc(m & row.q) === h)));
    assert.equal(attributes['data-cap'], String(cap)); assert.equal(attributes['data-step'], String(i + 1));
    assert.equal(attributes['data-query'], String(row.q)); assert.equal(attributes['data-hit'], String(row.h));
    assert.equal(attributes['data-before'], String(row.before)); assert.equal(attributes['data-after'], String(row.after));
    assert(attributes.class.split(/\s+/).includes(i < depth ? 'stacked' : 'not-stacked'));
    assert(attributes.class.includes('finished-sheet') === row.done);
    assert.equal(attributes.style, `--sheet:${colors[i]}`);
    assert(body.includes(`<b class="question-number">質問${i + 1}</b>`));
    const cells = [...body.matchAll(/<span class="id-position([^\"]*)"([^>]*)>(\d+)<\/span>/g)];
    assert.equal(cells.length, 11);
    cells.forEach((cell, j) => {
      const a = attrs(cell[2]), selected = row.q >> j & 1;
      assert.equal(cell[3], String(data.ids[j])); assert.equal(a['data-id'], String(data.ids[j]));
      assert.equal(a['data-selected'], String(selected)); assert.equal(cell[1].includes('investigate'), Boolean(selected));
      assert.equal(a.style, `--sheet:${colors[i]}`);
      result.id_cells_checked++;
    });
    if (row.done) {
      assert.equal(row.q, 0); assert.equal(row.h, 0); assert.equal(row.before, 1); assert.equal(row.after, 1);
      assert(body.includes('確定済み・追加質問不要')); assert(body.includes('誰も調べません'));
      result.finished_placeholder_rows_checked++;
    } else {
      const groupMarkup = body.match(/<div class="hit-counts">([\s\S]*?)<\/div>/)?.[1];
      assert(groupMarkup);
      for (let h = 0; h <= cap; h++) {
        const amount = row.groups[h] || 0;
        assert(groupMarkup.includes(`<span class="${h === row.h ? 'chosen-hit' : ''}">${h}人 → ${amount}${cap === 1 ? '候補' : '組'}</span>`));
      }
      assert.equal(Object.values(row.groups).reduce((sum, amount) => sum + amount, 0), row.before);
      assert(body.includes('候補数の計算の元を見る'));
    }
    assertCards(cards(body), enumerate(cap), rows, depth, i);
    result.question_rows_checked++;
    result.question_card_checks += enumerate(cap).length;
  });
  const kept = enumerate(cap).filter(candidate => signature(candidate, rows).slice(0, depth).split('').every(v => v === '1'));
  assert.deepEqual(sorted(viewer.json(`MIA_LESSON.remaining(${cap},${member},${depth})`)), sorted(kept));
  const merged = viewer.ids.get(`lesson-result-${cap}`).innerHTML;
  const combined = merged.match(/<div class="candidate-sheet combined-sheet"([^>]*)>/);
  assert(combined); assert.equal(attrs(combined[1])['data-remaining'], String(kept.length));
  assert.equal(attrs(combined[1])['data-depth'], String(depth));
  assertCards(cards(merged), enumerate(cap), rows, depth);
  result.combined_card_checks += enumerate(cap).length;
  if (depth === 5) {
    assert.deepEqual(kept, [member]);
    assert(merged.includes(`<b>残る1枚：ID ${data.ids.filter((_, i) => member >> i & 1).join('・')}</b>`));
  }
}

(function main() {
  const result = {
    status: 'passed', environment: 'Node VM with simulated DOM; independent saved-source enumeration and generated lesson markup',
    html_bytes: Buffer.byteLength(html), html_sha256: crypto.createHash('sha256').update(html).digest('hex'),
    one_member_examples: 0, two_member_examples: 0, question_rows_checked: 0, id_cells_checked: 0,
    question_card_checks: 0, combined_card_checks: 0, finished_placeholder_rows_checked: 0,
    formulas_checked: [], controls_checked: [], visual_browser_validation: false,
  };
  const viewer = simulatedViewer();
  assert(html.indexOf('id="lesson-questions-1"') < html.indexOf('id="focus"'));
  assert(html.indexOf('id="lesson-questions-2"') < html.indexOf('id="focus"'));
  result.controls_checked.push('one-member and two-member lesson precedes Team-specific controls');
  assert.deepEqual(viewer.json('MIA_LESSON.values()'), {depth: 5, masks: {'1': 64, '2': 1025}});
  const advancedStart = viewer.json('history');
  for (const cap of [1, 2]) {
    const options = [...viewer.ids.get(`lesson-example-${cap}`).innerHTML.matchAll(/<option\b[^>]*\bvalue="(\d+)"[^>]*>/g)].map(m => Number(m[1]));
    assert.deepEqual(sorted(options), enumerate(cap));
    for (const member of enumerate(cap)) {
      viewer.select(`lesson-example-${cap}`, member);
      assert.equal(viewer.json('MIA_LESSON.values()').masks[cap], member);
      assertColumn(viewer, cap, member, 5, result);
      result[cap === 1 ? 'one_member_examples' : 'two_member_examples']++;
    }
  }
  assert.equal(result.one_member_examples, 11); assert.equal(result.two_member_examples, 55);
  assert.deepEqual(viewer.json('history'), advancedStart);
  result.controls_checked.push('all 11 one-member and all 55 pair examples', 'lesson example selection leaves six-Team route unchanged');
  for (const [cap, member] of [[1, 64], [2, 1025]]) viewer.select(`lesson-example-${cap}`, member);
  for (let depth = 0; depth <= 5; depth++) {
    viewer.select('lesson-depth', depth);
    assert.equal(viewer.json('MIA_LESSON.values()').depth, depth);
    assert.equal(viewer.ids.get('lesson-back').disabled, depth === 0);
    assert.equal(viewer.ids.get('lesson-next').disabled, depth === 5);
    assertColumn(viewer, 1, 64, depth, result);
    assertColumn(viewer, 2, 1025, depth, result);
  }
  viewer.click('lesson-start'); assert.equal(viewer.json('MIA_LESSON.values()').depth, 0);
  viewer.click('lesson-back'); assert.equal(viewer.json('MIA_LESSON.values()').depth, 0);
  for (let i = 0; i < 5; i++) {viewer.click('lesson-next'); assert.equal(viewer.json('MIA_LESSON.values()').depth, i + 1);}
  viewer.click('lesson-next'); assert.equal(viewer.json('MIA_LESSON.values()').depth, 5);
  viewer.click('lesson-back'); assert.equal(viewer.json('MIA_LESSON.values()').depth, 4);
  viewer.click('lesson-end'); assert.equal(viewer.json('MIA_LESSON.values()').depth, 5);
  assert.equal(viewer.ids.get('lesson-depth').value, '5');
  assert.deepEqual(viewer.json('history'), advancedStart);
  result.controls_checked.push('all zero through five sheet depths', 'start/back/next/end controls and depth bounds', 'matched/rejected/unused color badges');
  const oneMarkup = viewer.ids.get('lesson-questions-1').innerHTML;
  const pairMarkup = viewer.ids.get('lesson-questions-2').innerHTML;
  assert(oneMarkup.includes('選択した5人 → 1人の回答なら5候補。選ばない6人 → 0人なら6候補。'));
  assert(pairMarkup.includes('0人：7×6÷2＝21組 ／ 1人：4×7＝28組 ／ 2人：4×3÷2＝6組。合計55組。'));
  assert.equal(11 * 10 / 2, enumerate(2).length);
  assert.equal(7 * 6 / 2 + 4 * 7 + 4 * 3 / 2, enumerate(2).length);
  result.formulas_checked.push('11 choose 2 = 11×10÷2 = 55', 'first one-member split 6+5=11', 'first pair split 21+28+6=55', 'later counts enumerate previously surviving cards');
  if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_lesson_ui.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
})();
