'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const {simulatedViewer} = require('./test_fixed_ui.cjs');
const root = path.resolve(__dirname, '../..'), html = fs.readFileSync(path.join(root, 'docs/visualizer/MIA_5query_cards.html'), 'utf8');
const history = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/resources/intro_history.json'), 'utf8')), c = history.conditions;
const close = (a, b) => assert(Math.abs(a - b) < 1e-13, `${a} != ${b}`);
const value = ratio => {const [n, d] = ratio.split('/').map(Number); return n / d;};
// Independently sort 24 labels, group confidence ties, then interpolate the ROC at FPR .01.
function roc(m, n) {
  const rows = [...Array.from({length: m}, () => [.4, 1]), ...Array.from({length: n - m}, () => [.4, 0]), ...Array.from({length: 24 - n}, () => [.1, 0])].sort((a, b) => b[0] - a[0]);
  const points = [[0, 0]]; let tp = 0, fp = 0;
  rows.forEach((row, i) => {tp += row[1]; fp += 1 - row[1]; if (i === rows.length - 1 || row[0] !== rows[i + 1][0]) points.push([fp / (24 - m), tp / m]);});
  const i = points.findIndex(p => p[0] >= .01), a = points[i - 1], b = points[i];
  return a[1] + (.01 - a[0]) / (b[0] - a[0]) * (b[1] - a[1]);
}
function main() {
  const v = simulatedViewer(), original = v.json('MIA_FIXED_LESSON.snapshot()'), operational = v.json('MIA_FIXED_VIEWER.snapshot()');
  const report = {status: 'passed', environment: 'Node simulated DOM; independent 24-label tied-score ROC interpolation', editable_configurations: 0, team_input_changes: 0, saved_configurations_loaded: 0, zero_active_mean_undefined: true, eleven_member_limit_checked: true, custom_counts_preserved_across_stages: true, manual_controls_have_no_timers: true, visual_browser_validation: false, html_bytes: Buffer.byteLength(html), html_sha256: crypto.createHash('sha256').update(html).digest('hex')};
  const zero = () => Object.fromEntries(c.teams.map(t => [t, 0]));
  const unchanged = () => {assert.equal(v.timers.length, 0); assert.deepEqual(v.json('MIA_FIXED_LESSON.snapshot()'), original); assert.deepEqual(v.json('MIA_FIXED_VIEWER.snapshot()'), operational);};
  const model = () => v.json('MIA_FIXED_INTRO.historyModel()');
  const check = counts => {
    const state = model(), active = c.teams.filter(t => counts[t] > 0), total = Object.values(counts).reduce((a, b) => a + b, 0);
    assert.deepEqual(state.counts, counts); assert.equal(state.activeTeams, active.length); assert.equal(state.totalMembers, total); assert.equal(state.valid, total <= 11); assert.equal(state.defined, active.length > 0);
    for (const t of c.teams) {assert.equal(v.elements.get(`history-count-${t}`).value, String(counts[t])); assert.equal(v.elements.get(`history-count-cell-${t}`).attributes['data-included'], String(counts[t] > 0)); assert.equal(v.elements.get(`history-count-status-${t}`).textContent, counts[t] ? '平均に入る' : '平均から外す');}
    for (const round of [1, 2]) {
      const prediction = state.predictions[round - 1], sizes = c[round === 1 ? 'round1_sizes' : 'round2_sizes'];
      assert.equal(prediction.denominator, active.length); assert.equal(prediction.excluded, 22 - active.length); assert.deepEqual(prediction.contributions.map(x => x.team), active);
      const sum = active.reduce((sum, t) => sum + roc(counts[t], sizes[c.teams.indexOf(t)]), 0);
      close(value(prediction.sumExact), sum);
      if (active.length) {close(prediction.meanValue, sum / active.length); close(value(prediction.meanExact), sum / active.length); close(state.predicted[round - 1], sum / active.length);}
      else {assert.equal(prediction.meanExact, null); assert.equal(prediction.meanValue, null); assert.equal(state.predicted[round - 1], null);}
      const markup = v.elements.get(`history-live-mean-${round}`).innerHTML;
      assert(!/NaN|Infinity/.test(markup)); if (active.length) {assert(markup.includes(`${active.length}チーム`)); assert(markup.includes(prediction.meanExact));} else assert(markup.includes('未定義'));
      const expectedCheck = total > 11 ? '条件外' : !active.length ? '未定義' : state.observed[round - 1] === null ? '未表示' : Math.abs(prediction.meanValue - state.observed[round - 1]) < c.score_match_tolerance ? '一致' : '不一致';
      assert.equal(state.checks[round - 1], expectedCheck); assert.equal(v.elements.get(`history-live-mean-${round}`).attributes['data-check'], expectedCheck);
    }
    assert.equal(v.elements.get('history-count-summary').attributes['data-valid'], String(total <= 11)); unchanged(); report.editable_configurations++;
  };
  v.click('intro-tab-history'); assert.equal(v.elements.get('history-editor-panel').hidden, false); assert.equal(v.elements.get('history-mean-panel').hidden, true); check(model().counts);
  v.click('history-count-zero'); check(zero()); assert.equal(model().source, 'custom'); assert.equal(model().caseID, null); assert(v.elements.get('history-count-summary').innerHTML.includes('K＝0'));
  v.click('history-stage-1'); v.click('history-stage-2'); check(zero()); assert.equal(v.elements.get('history-editor-panel').hidden, false); assert.deepEqual(model().observed, [c.round1_rare_score, c.round2_rare_score]);
  v.click('history-show-mean'); assert.equal(v.elements.get('history-mean-panel').hidden, false); assert(v.elements.get('history-mean-equation').innerHTML.includes('未定義')); assert(!/NaN|Infinity/.test(v.elements.get('history-mean-equation').innerHTML)); v.click('history-show-editor');
  for (const t of c.teams) for (const m of [0, 1, 2]) {
    v.click('history-count-zero'); v.select(`history-count-${t}`, m); const counts = {...zero(), [t]: m}; check(counts); assert.equal(model().team, t); assert.equal(v.elements.get('history-team').value, t); report.team_input_changes++;
  }
  // A second person changes TPR while K stays 1; a second occupied Team changes K to 2.
  v.click('history-count-zero'); v.select('history-count-6', 1); const one = model(); v.select('history-count-6', 2); const two = model(); assert.equal(one.activeTeams, two.activeTeams); assert.notEqual(one.predicted[0], two.predicted[0]); assert.notEqual(one.predicted[1], two.predicted[1]);
  v.select('history-count-10', 1); assert.equal(model().activeTeams, 2); v.select('history-count-6', 0); assert.equal(model().activeTeams, 1); check(model().counts);
  // Exercise legal count distributions beyond the saved 102 cases, including all 44 single-Team cases.
  for (let seed = 0; seed < 32; seed++) {const counts = zero(); let remaining = 1 + seed % 11; for (let j = 0; j < 22 && remaining; j++) {const t = c.teams[(seed + j * 7) % 22], m = Math.min(remaining, 1 + (seed + j) % 2); counts[t] = m; remaining -= m;} v.execute(`MIA_FIXED_INTRO.setCounts(${JSON.stringify(counts)})`); check(counts);}
  const eleven = zero(); c.teams.slice(0, 5).forEach(t => eleven[t] = 2); eleven[c.teams[5]] = 1; v.execute(`MIA_FIXED_INTRO.setCounts(${JSON.stringify(eleven)})`); check(eleven); assert.equal(model().valid, true);
  v.select(`history-count-${c.teams[5]}`, 2); check({...eleven, [c.teams[5]]: 2}); assert.equal(model().valid, false); assert(model().checks.every(x => x === '条件外')); assert(v.elements.get('history-count-summary').innerHTML.includes('合計11人を超え'));
  const all = Object.fromEntries(c.teams.map(t => [t, 2])); v.execute(`MIA_FIXED_INTRO.setCounts(${JSON.stringify(all)})`); check(all); assert.equal(model().activeTeams, 22); assert.equal(model().totalMembers, 44);
  v.click('history-stage-0'); check(all); v.click('intro-next'); check(all); v.click('intro-next'); check(all); assert.equal(v.elements.get('history-editor-panel').hidden, false);
  v.click('history-count-winner'); const winner = history.cases.find(x => x.case_id === history.winner); check(winner.counts); assert.equal(model().source, 'saved'); assert.deepEqual(model().checks, ['一致', '一致']); assert.equal(model().activeTeams, 6); assert.equal(model().totalMembers, 9);
  for (const row of history.cases) {v.select('history-case', row.case_id); check(row.counts); assert.equal(model().source, 'saved'); assert.equal(model().caseID, row.case_id); report.saved_configurations_loaded++;}
  // Explicit editing view keeps a saved comparison on stage changes, rather than replacing it with N015.
  v.select('history-case', 'N001'); v.click('history-stage-1'); v.click('history-stage-2'); assert.equal(model().caseID, 'N001'); assert.equal(v.elements.get('history-editor-panel').hidden, false);
  const before = model(); assert.throws(() => v.execute('MIA_FIXED_INTRO.setCount("6",3)'), /所属人数/); assert.throws(() => v.execute('MIA_FIXED_INTRO.setCount("99",1)'), /所属人数/); assert.throws(() => v.execute('MIA_FIXED_INTRO.setCounts({"6":1})'), /全22チーム/); assert.throws(() => v.execute('MIA_FIXED_INTRO.setCounts(null)'), /全22チーム/); assert.deepEqual(model(), before); unchanged();
  if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_fixed_counts.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
}
main();
