#!/usr/bin/env node
'use strict';
// Independently enumerate fixed question cards and inspect generated controls.
// A simulated DOM does not provide browser layout or visual verification.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const htmlPath = path.join(root, 'docs/visualizer/MIA_5query_cards.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const dataScript = scripts.find(script => /type="application\/json"/.test(script[1]));
assert(dataScript, 'An embedded data payload is required');
const data = JSON.parse(dataScript[2]);
const IDS = [47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675];
const QUERIES = [1330, 1069, 678, 87, 779];
const CAPS = [1, 2, 1, 2, 1, 2];
const DEMO = [128, 3, 4, 24, 32, 320];
const COLORS = ['#2476b8', '#008576', '#bc6913', '#a4497a', '#6552ad'];
const report = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/size_report_5query.json'), 'utf8'));
const sourceCode = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/nonadaptive_code.json'), 'utf8'));
const encoding = JSON.parse(fs.readFileSync(path.join(root, 'analysis/full_depth4_20261007/universal_score_encoding.json'), 'utf8'));
assert.deepEqual(data.ids, IDS); assert.deepEqual(data.query_masks, QUERIES); assert.deepEqual(data.caps, CAPS);
assert.deepEqual(data.demo_masks, DEMO); assert.deepEqual(data.encoding, encoding);
assert.deepEqual(data.codewords, [26, 29, 14, 18, 9, 7, 8, 4, 17, 20, 3]);
assert.deepEqual(data.provenance, sourceCode.provenance); assert.equal(data.provenance.kind, 'new_construction');
assert.equal(report.strategy, 'nonadaptive'); assert.deepEqual(report.single_team_max_depth, [5, 5]);
assert.equal(Buffer.byteLength(html), report.html_bytes);
assert.equal(crypto.createHash('sha256').update(html).digest('hex'), report.html_sha256);
for (const source of report.source_files) {
  const bytes = fs.readFileSync(path.join(root, source.path));
  assert.equal(bytes.length, source.bytes); assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), source.sha256);
}
const adaptive = fs.readFileSync(path.join(root, 'docs/visualizer/MIA_5query_adaptive.html'));
assert.equal(adaptive.length, 67765);
assert.equal(crypto.createHash('sha256').update(adaptive).digest('hex'), 'ce3284703dd06f457ff556d63f366473992059bf539901c01c4016f1ac746d8a');
assert.equal(scripts.length, 4);
assert.equal(scripts[0][2], fs.readFileSync(path.join(__dirname, 'tree_core.js'), 'utf8'));
assert.equal(scripts[1][2], fs.readFileSync(path.join(__dirname, 'fixed_core.js'), 'utf8'));
const pc = mask => mask.toString(2).replace(/0/g, '').length;
const enumerate = cap => Array.from({length: 2047}, (_, i) => i + 1).filter(mask => pc(mask) === cap);
const sorted = values => values.slice().sort((a, b) => a - b);
const attrs = markup => Object.fromEntries([...markup.matchAll(/\b([\w-]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));
const label = mask => IDS.filter((_, index) => mask >> index & 1).join('・');
const signature = member => QUERIES.map(query => pc(member & query));
const candidateSignature = (candidate, hits) => QUERIES.map((query, index) => pc(candidate & query) === hits[index] ? '1' : '0').join('');
function independentTrace(cap, member) {
  assert.equal(pc(member), cap);
  const hits = signature(member), rows = [];
  let states = enumerate(cap);
  for (let step = 0; step < 5; step++) {
    const query = QUERIES[step], before = states.slice();
    const groups = Array.from({length: cap + 1}, (_, h) => before.filter(candidate => pc(candidate & query) === h));
    states = groups[hits[step]].slice();
    rows.push({step, query, hit: hits[step], before, after: states.slice(), groups});
  }
  assert.deepEqual(states, [member]);
  return rows;
}
function independentSets(observations) {
  return CAPS.map((cap, team) => enumerate(cap).filter(member => observations.every((hits, round) => hits === null || pc(member & QUERIES[round]) === hits[team])));
}
function hasDisjoint(sets) {
  const order = sets.map((set, team) => ({set, team})).sort((a, b) => a.set.length - b.set.length);
  const failed = new Set();
  function search(index, used) {
    if (index === 6) return true;
    const key = `${index}:${used}`; if (failed.has(key)) return false;
    for (const member of order[index].set) if (!(used & member) && search(index + 1, used | member)) return true;
    failed.add(key); return false;
  }
  return search(0, 0);
}
const code = hits => hits.reduce((sum, hit, team) => sum + hit * 3 ** team, 0);
function exactMean(hits) {
  const numerator = encoding.integer_values[String(code(hits))], denominator = encoding.mean_denominator;
  const gcd = (a, b) => b ? gcd(b, a % b) : a, divisor = gcd(numerator, denominator);
  return `${numerator / divisor}/${denominator / divisor}`;
}

function simulatedViewer() {
  const elements = new Map(), downloads = [], blobs = new Map();
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
      this.selects = [...this._html.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/g)].map(m => {
        const el = new Element('select'), a = attrs(m[1]);
        for (const [key, value] of Object.entries(a)) if (key.startsWith('data-')) el.dataset[key.slice(5)] = value;
        el.innerHTML = m[2]; return el;
      });
    }
    get innerHTML() { return this._html; }
    replaceChildren(...children) { this.children = children; this._html = ''; this.selects = []; }
    append(...children) { this.children.push(...children); }
    querySelectorAll(selector) {
      const match = /^\[data-([\w-]+)\]$/.exec(selector); assert(match, `Unsupported selector ${selector}`);
      return (this.selects || []).filter(el => Object.hasOwn(el.dataset, match[1]));
    }
    click() {
      if (this.disabled) return;
      if (this.tagName === 'A') { downloads.push({name: this.download, blob: blobs.get(this.href)}); return; }
      return this.onclick?.();
    }
  }
  for (const m of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) elements.set(m[2], new Element(m[1], m[2]));
  const dataID = attrs(dataScript[1]).id; assert(dataID); elements.get(dataID).textContent = dataScript[2];
  const location = {hash: '', pathname: '/MIA_5query_cards.html', search: ''};
  const document = {getElementById(id) {const el = elements.get(id); assert(el, `Unknown DOM id ${id}`); return el;}, createElement(tag) {return new Element(tag);}};
  const sandbox = {document, location, history: {replaceState(_state, _title, url) {const i = url.indexOf('#'); location.hash = i < 0 ? '' : url.slice(i);}},
    Blob, URL: {createObjectURL(blob) {const key = `blob:test-${blobs.size}`; blobs.set(key, blob); return key;}, revokeObjectURL() {}},
    setTimeout(callback) {queueMicrotask(callback); return 0;}, console};
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  scripts.filter(script => script !== dataScript).forEach((script, i) => vm.runInContext(script[2], context, {filename: `fixed-page-script-${i}.js`}));
  const execute = code => vm.runInContext(code, context);
  const json = code => JSON.parse(execute(`JSON.stringify(${code})`));
  const select = (id, value) => {const el = elements.get(id); assert(el); el.value = String(value); assert(el.onchange, `No change handler on ${id}`); el.onchange();};
  const click = id => {const el = elements.get(id); assert(el); el.click();};
  return {elements, execute, json, select, click, downloads};
}

function cards(markup) {
  const starts = [...markup.matchAll(/<div class="lesson-member([^"]*)"([^>]*)>/g)];
  return starts.map((m, i) => ({classes: m[1], attrs: attrs(m[2]), body: markup.slice(m.index + m[0].length, i + 1 < starts.length ? starts[i + 1].index : markup.length)}));
}
function assertCards(actual, cap, hits, depth, result, question = null) {
  assert.equal(actual.length, enumerate(cap).length);
  assert.deepEqual(sorted(actual.map(card => Number(card.attrs['data-member']))), enumerate(cap));
  for (const card of actual) {
    const member = Number(card.attrs['data-member']), sig = candidateSignature(member, hits);
    const pass = question === null ? sig.slice(0, depth).split('').every(value => value === '1') : sig[question] === '1';
    assert.equal(card.attrs['data-signature'], sig); assert.equal(card.attrs['data-pass'], pass ? '1' : '0');
    assert(card.classes.includes(pass ? 'fits' : 'blocked'));
    assert(card.body.startsWith(`<span>${IDS.filter((_, i) => member >> i & 1).join('<br>')}</span>`));
    if (question === null) {
      const badges = [...card.body.matchAll(/<i class="condition ([^"]*)"([^>]*)>([1-5])<\/i>/g)];
      assert.equal(badges.length, 5);
      badges.forEach((badge, i) => {
        assert.equal(badge[1], i >= depth ? 'unused' : sig[i] === '1' ? 'agree' : 'reject');
        assert.equal(attrs(badge[2]).style, `--sheet:${COLORS[i]}`); assert.equal(badge[3], String(i + 1));
      });
      result.combined_card_checks++;
    } else result.question_card_checks++;
  }
}

function assertColumn(viewer, cap, member, depth, result) {
  const rows = independentTrace(cap, member), hits = signature(member);
  const apiRows = viewer.json(`MIA_FIXED_LESSON.trace(${cap},${member})`);
  assert.equal(apiRows.length, 5);
  const questionMarkup = viewer.elements.get(`lesson-questions-${cap}`).innerHTML;
  const questions = [...questionMarkup.matchAll(/<article\b([^>]*)>([\s\S]*?)<\/article>/g)];
  assert.equal(questions.length, 5);
  questions.forEach((question, index) => {
    const row = rows[index], api = apiRows[index], a = attrs(question[1]), body = question[2];
    assert.equal(api.step, index); assert.equal(api.query, row.query); assert.equal(api.hit, row.hit);
    assert.equal(api.complete, false); assert.equal(api.alreadyUnique, row.before.length === 1);
    assert.deepEqual(sorted(api.before), row.before); assert.deepEqual(sorted(api.after), row.after);
    for (let hit = 0; hit <= cap; hit++) assert.deepEqual(sorted(api.groups[hit]), row.groups[hit]);
    assert.equal(a['data-cap'], String(cap)); assert.equal(a['data-step'], String(index + 1));
    assert.equal(a['data-query'], String(row.query)); assert.equal(a['data-hit'], String(row.hit));
    assert.equal(a['data-before'], String(row.before.length)); assert.equal(a['data-after'], String(row.after.length));
    assert.equal(a.class.includes('finished-sheet'), false);
    assert(a.class.split(/\s+/).includes(index < depth ? 'stacked' : 'not-stacked'));
    assert.equal(a.style, `--sheet:${COLORS[index]}`);
    assert(body.includes(`<b class="question-number">質問${index + 1}</b>`));
    assert(!body.includes('追加質問不要')); assert(!body.includes('誰も調べません'));
    const cells = [...body.matchAll(/<span class="id-position([^"]*)"([^>]*)>(\d+)<\/span>/g)];
    assert.equal(cells.length, 11);
    cells.forEach((cell, idIndex) => {
      const ca = attrs(cell[2]), selected = row.query >> idIndex & 1;
      assert.equal(cell[3], String(IDS[idIndex])); assert.equal(ca['data-id'], String(IDS[idIndex]));
      assert.equal(ca['data-selected'], String(selected)); assert.equal(cell[1].includes('investigate'), Boolean(selected));
      assert.equal(ca.style, `--sheet:${COLORS[index]}`); result.id_cells_checked++;
    });
    assert.equal(cells.filter(cell => cell[1].includes('investigate')).length, 5);
    const groupMarkup = body.match(/<div class="hit-counts">([\s\S]*?)<\/div>/)?.[1]; assert(groupMarkup);
    for (let hit = 0; hit <= cap; hit++) assert(groupMarkup.includes(`<span class="${hit === row.hit ? 'chosen-hit' : ''}">${hit}人 → ${row.groups[hit].length}${cap === 1 ? '候補' : '組'}</span>`));
    assert.equal(row.groups.reduce((sum, group) => sum + group.length, 0), row.before.length);
    assert(body.includes('候補数の計算の元を見る'));
    // A single transparent sheet ignores the earlier sheets; its color is
    // checked over all original cards, while displayed numbers use survivors.
    assertCards(cards(body), cap, hits, depth, result, index);
    result.question_rows_checked++;
    if (row.before.length === 1) result.fixed_questions_after_early_certainty++;
  });
  const kept = enumerate(cap).filter(candidate => candidateSignature(candidate, hits).slice(0, depth).split('').every(value => value === '1'));
  assert.deepEqual(sorted(viewer.json(`MIA_FIXED_LESSON.remaining(${cap},${member},${depth})`)), kept);
  const merged = viewer.elements.get(`lesson-result-${cap}`).innerHTML;
  const combined = merged.match(/<div class="candidate-sheet combined-sheet"([^>]*)>/); assert(combined);
  assert.equal(attrs(combined[1])['data-remaining'], String(kept.length)); assert.equal(attrs(combined[1])['data-depth'], String(depth));
  assertCards(cards(merged), cap, hits, depth, result);
  assert.equal(viewer.elements.get(`lesson-counts-${cap}`).textContent, [enumerate(cap).length, ...rows.map(row => row.after.length)].join(' → '));
  if (depth === 5) {assert.deepEqual(kept, [member]); assert(merged.includes(`<b>残る1枚：ID ${label(member)}</b>`));}
}

async function main() {
  const result = {status: 'passed', environment: 'Node VM with simulated DOM; independent enumeration of fixed questions and rendered card markup',
    html_bytes: Buffer.byteLength(html), html_sha256: crypto.createHash('sha256').update(html).digest('hex'),
    one_member_examples: 0, two_member_examples: 0, example_prefixes_checked: 0, question_rows_checked: 0,
    id_cells_checked: 0, question_card_checks: 0, combined_card_checks: 0, fixed_questions_after_early_certainty: 0,
    formulas_checked: [], controls_checked: [], mean_decodes_checked: [], negative_cases: [], visual_browser_validation: false};
  const viewer = simulatedViewer();
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.values()'), {depth: 5, masks: {'1': 128, '2': 3}});
  for (const cap of [1, 2]) {
    const options = [...viewer.elements.get(`lesson-example-${cap}`).innerHTML.matchAll(/<option\b[^>]*\bvalue="(\d+)"[^>]*>/g)].map(m => Number(m[1]));
    assert.deepEqual(sorted(options), enumerate(cap));
    for (const member of enumerate(cap)) {
      viewer.select(`lesson-example-${cap}`, member);
      assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[cap], member);
      for (let depth = 0; depth <= 5; depth++) {
        viewer.select('lesson-depth', depth);
        assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, depth);
        assertColumn(viewer, cap, member, depth, result); result.example_prefixes_checked++;
      }
      result[cap === 1 ? 'one_member_examples' : 'two_member_examples']++;
    }
  }
  assert.equal(result.one_member_examples, 11); assert.equal(result.two_member_examples, 55);
  assert.equal(result.example_prefixes_checked, 396);
  assert(result.fixed_questions_after_early_certainty > 0);
  result.controls_checked.push('all 11 single and 55 pair examples at every prefix zero through five',
    'same five nonzero query masks and same five selected IDs for every example and prefix',
    'all original cards keep fixed positions across sheets',
    'question card colors use only that sheet; counts use earlier surviving cards',
    'combined candidate cards and all matched/rejected/unused badges',
    'fixed questions retained after early certainty');
  for (const [cap, member] of [[1, 128], [2, 3]]) viewer.select(`lesson-example-${cap}`, member);
  viewer.click('lesson-start'); assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, 0);
  assert(viewer.elements.get('lesson-back').disabled); assert(!viewer.elements.get('lesson-next').disabled);
  viewer.click('lesson-back'); assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, 0);
  for (let index = 0; index < 5; index++) {viewer.click('lesson-next'); assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, index + 1);}
  assert(viewer.elements.get('lesson-next').disabled);
  viewer.click('lesson-next'); assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, 5);
  viewer.click('lesson-back'); assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, 4);
  viewer.click('lesson-end'); assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth, 5);
  assert.equal(viewer.elements.get('lesson-depth').value, '5');
  result.controls_checked.push('start/back/next/end controls and depth bounds');
  const singleMarkup = viewer.elements.get('lesson-questions-1').innerHTML, pairMarkup = viewer.elements.get('lesson-questions-2').innerHTML;
  assert(singleMarkup.includes('選択した5人 → 1人の回答なら5候補。選ばない6人 → 0人なら6候補。'));
  assert(pairMarkup.includes('0人：6×5÷2＝15組 ／ 1人：5×6＝30組 ／ 2人：5×4÷2＝10組。合計55組。'));
  assert.equal(11 * 10 / 2, enumerate(2).length); assert.equal(6 * 5 / 2 + 5 * 6 + 5 * 4 / 2, enumerate(2).length);
  result.formulas_checked.push('11 choose 2 = 11×10÷2 = 55', 'fixed first-question one-member split 6+5=11',
    'fixed first-question pair split 15+30+10=55', 'every later count enumerates earlier surviving masks');
  const operationSnapshot = () => viewer.json('MIA_FIXED_VIEWER.snapshot()');
  const initialOperation = operationSnapshot();
  assert.deepEqual(initialOperation, {observations: Array(5).fill(null), meanInputs: Array(5).fill(null), counts: [11, 55, 11, 55, 11, 55], queries: QUERIES});
  const tableQuestions = [...html.matchAll(/<tr data-fixed-round="(\d+)" data-query="(\d+)"><td>\d+<\/td><td>([^<]+)<\/td>/g)];
  assert.equal(tableQuestions.length, 5);
  tableQuestions.forEach((row, round) => {assert.equal(Number(row[1]), round + 1); assert.equal(Number(row[2]), QUERIES[round]); assert.equal(row[3], label(QUERIES[round]));});
  for (let team = 0; team < 6; team++) {
    const count = data.high[team] - 5;
    assert.equal(data.padding_counts[team], count);
    assert(html.includes(`<tr><td>${data.teams[team]}</td><td>5人</td><td>${data.negative_ids.slice(0, count).join('・')}（${count}人）</td><td>${data.high[team]}人</td></tr>`));
  }
  result.controls_checked.push('embedded source code and score encoding match source files',
    'preserved adaptive HTML is byte-identical to the previous 67,765-byte version',
    'six-Team fixed question table and constant padding IDs/counts');

  const demoResponses = QUERIES.map(query => DEMO.map(member => pc(member & query)));
  const demoMeans = ['73/6750', '839/54000', '577/27000', '31/1800', '13/1200'];
  assert.deepEqual(demoResponses.map(exactMean), demoMeans);
  const observations = Array(5).fill(null);
  for (let round = 0; round < 5; round++) {
    observations[round] = demoResponses[round];
    viewer.elements.get(`fixed-mean-${round + 1}`).value = demoMeans[round]; viewer.click(`fixed-decode-${round + 1}`);
    assert(viewer.elements.get('fixed-error').hidden);
    const snapshot = operationSnapshot();
    assert.deepEqual(snapshot.observations, observations); assert.deepEqual(snapshot.queries, QUERIES);
    assert.deepEqual(snapshot.counts, independentSets(observations).map(set => set.length));
    assert(viewer.elements.get(`fixed-answer-${round + 1}`).textContent.includes(`10:${demoResponses[round][1]}人`));
    assert(viewer.elements.get('fixed-status').textContent.includes(`${round + 1}/5回`));
    result.mean_decodes_checked.push(demoMeans[round]);
  }
  assert.deepEqual(operationSnapshot().counts, [1, 1, 1, 1, 1, 1]);
  assert(viewer.elements.get('fixed-results').innerHTML.includes('全6 Teamの所属が1候補ずつ'));
  assert(viewer.elements.get('fixed-results').innerHTML.includes('対象6 Team外：495・675'));
  const beforeLessonChange = operationSnapshot();
  viewer.select('lesson-example-1', 1); viewer.select('lesson-example-2', 1025); viewer.click('lesson-start');
  assert.deepEqual(operationSnapshot(), beforeLessonChange);
  result.controls_checked.push('all five exact fictional demo means', 'lesson selection leaves operational observations unchanged');

  viewer.click('fixed-export');
  const download = viewer.downloads.at(-1); assert.equal(download.name, 'MIA_nonadaptive5_result.json');
  const exported = JSON.parse(await download.blob.text());
  assert.equal(exported.strategy, 'nonadaptive'); assert.deepEqual(exported.fixed_query_masks, QUERIES);
  assert.deepEqual(exported.query_ids, QUERIES.map(query => IDS.filter((_, i) => query >> i & 1)));
  assert.deepEqual(exported.responses, observations); assert.deepEqual(exported.remaining_counts, [1, 1, 1, 1, 1, 1]);
  for (let team = 0; team < 6; team++) assert.deepEqual(exported.members_by_team[data.teams[team]], IDS.filter((_, i) => DEMO[team] >> i & 1));
  result.controls_checked.push('JSON export of unchanged questions, observations, and actual ID results');

  viewer.click('fixed-clear'); assert.deepEqual(operationSnapshot(), initialOperation);
  for (let round = 4; round >= 0; round--) {
    viewer.elements.get(`fixed-mean-${round + 1}`).value = demoMeans[round]; viewer.click(`fixed-decode-${round + 1}`);
    assert(viewer.elements.get('fixed-error').hidden); assert.deepEqual(operationSnapshot().queries, QUERIES);
    assert.equal(operationSnapshot().observations.filter(hits => hits !== null).length, 5 - round);
  }
  assert.deepEqual(operationSnapshot().observations, observations);
  viewer.click('fixed-clear'); viewer.click('fixed-demo');
  assert.deepEqual(operationSnapshot().observations, observations); assert.deepEqual(operationSnapshot().counts, [1, 1, 1, 1, 1, 1]);
  assert.deepEqual(operationSnapshot().queries, QUERIES);
  result.controls_checked.push('clear control', 'reverse-order mean entry with empty intervening rows', 'all-five demo shortcut');

  viewer.click('fixed-clear');
  const alphabet = [];
  for (let n = 0; n < 729; n++) {
    const hits = CAPS.map((_, team) => Math.floor(n / 3 ** team) % 3);
    if (hits.every((hit, team) => hit <= CAPS[team])) alphabet.push(hits);
  }
  assert.equal(alphabet.length, 216);
  result.exact_mean_round_cases = 0; result.feasible_mean_round_cases = 0; result.infeasible_mean_round_cases = 0;
  for (let round = 0; round < 5; round++) for (const hits of alphabet) {
    viewer.click('fixed-clear');
    const next = Array(5).fill(null); next[round] = hits;
    const expectedSets = independentSets(next), feasible = hasDisjoint(expectedSets);
    // Every round selects five candidates, so nine assigned people require
    // at least three and at most five selected positives across the Teams.
    const sum = hits.reduce((a, b) => a + b, 0); assert.equal(feasible, sum >= 3 && sum <= 5);
    viewer.elements.get(`fixed-mean-${round + 1}`).value = exactMean(hits); viewer.click(`fixed-decode-${round + 1}`);
    assert.equal(viewer.elements.get('fixed-error').hidden, feasible);
    const snapshot = operationSnapshot(); assert.deepEqual(snapshot.queries, QUERIES);
    if (feasible) {assert.deepEqual(snapshot.observations, next); assert.deepEqual(snapshot.counts, expectedSets.map(set => set.length)); result.feasible_mean_round_cases++;}
    else {assert.deepEqual(snapshot, initialOperation); assert(viewer.elements.get('fixed-error').textContent.includes('所属配置が成立しません')); result.infeasible_mean_round_cases++;}
    result.exact_mean_round_cases++;
  }
  assert.equal(result.exact_mean_round_cases, 1080);
  result.controls_checked.push('all 216 exact alphabet means at each of five fixed rows', 'joint feasibility rejection leaves observations and fixed questions unchanged');
  viewer.click('fixed-clear');
  const firstValues = Object.values(encoding.integer_values).sort((a, b) => a - b);
  for (const value of [`${firstValues[0] + firstValues[1]}/${encoding.mean_denominator * 2}`, '1', '-1', 'not-a-number']) {
    viewer.elements.get('fixed-mean-1').value = value; viewer.click('fixed-decode-1');
    assert(!viewer.elements.get('fixed-error').hidden); assert.deepEqual(operationSnapshot(), initialOperation);
  }
  result.negative_cases.push('exact midpoint score', 'above score range', 'below score range', 'invalid numeric input');
  viewer.click('fixed-demo'); const beforeContradiction = operationSnapshot();
  viewer.elements.get('fixed-mean-1').value = '0'; viewer.click('fixed-decode-1');
  assert(!viewer.elements.get('fixed-error').hidden); assert.deepEqual(operationSnapshot(), beforeContradiction);
  result.negative_cases.push('contradictory mean cannot replace a valid observation');
  if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_fixed_ui.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
}
main().catch(error => {console.error(error); process.exitCode = 1;});
