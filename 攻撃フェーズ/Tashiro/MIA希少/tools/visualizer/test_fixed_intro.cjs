'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const {simulatedViewer} = require('./test_fixed_ui.cjs');
const root = path.resolve(__dirname, '../..'), html = fs.readFileSync(path.join(root, 'docs/visualizer/MIA_5query_cards.html'), 'utf8');
const source = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/resources/intro_history.json'), 'utf8'));
const c = source.conditions;
const close = (a, b, tolerance = 1e-13) => assert(Math.abs(a - b) < tolerance, `${a} != ${b}`);
// Independent ROC construction: sort all 24 records, group tied scores, and interpolate.
function independentROC(m, n, h) {
  const rows = [...Array.from({length: h}, () => [0.4, 1]), ...Array.from({length: n - h}, () => [0.4, 0]),
    ...Array.from({length: m - h}, () => [0.1, 1]), ...Array.from({length: 24 - m - n + h}, () => [0.1, 0])];
  rows.sort((a, b) => b[0] - a[0]);
  const points = [[0, 0]]; let tp = 0, fp = 0;
  for (let i = 0; i < rows.length; i++) {
    tp += rows[i][1]; fp += 1 - rows[i][1];
    if (i === rows.length - 1 || rows[i][0] !== rows[i + 1][0]) points.push([fp / (24 - m), tp / m]);
  }
  const i = points.findIndex(p => p[0] >= .01), a = points[i - 1], b = points[i];
  return {points, score: a[1] + (.01 - a[0]) / (b[0] - a[0]) * (b[1] - a[1])};
}
function independentMean(row, round) {
  const sizes = c[round === 1 ? 'round1_sizes' : 'round2_sizes'];
  const scores = c.teams.map((t, i) => row.counts[t] ? independentROC(row.counts[t], sizes[i], row.counts[t]).score : null).filter(x => x !== null);
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}
async function main() {
  const v = simulatedViewer(), original = v.json('MIA_FIXED_LESSON.snapshot()'), observations = v.json('MIA_FIXED_VIEWER.snapshot()');
  const report = {status: 'passed', environment: 'Node simulated DOM; independent tied-score ROC interpolation over 24 records; all 102 saved count configurations',
    roc_configurations: 0, historical_configurations: 0, historical_team_settings: 0, bridge_score_cases: 0, manual_navigation_has_no_timers: true, visual_browser_validation: false,
    html_bytes: Buffer.byteLength(html), html_sha256: crypto.createHash('sha256').update(html).digest('hex')};
  assert.equal(v.json('MIA_FIXED_OVERVIEW.snapshot()').view, 'intro');assert.equal(v.json('MIA_FIXED_INTRO.snapshot()').page, 'roc');
  function unchanged() {assert.deepEqual(v.json('MIA_FIXED_LESSON.snapshot()'), original);assert.deepEqual(v.json('MIA_FIXED_VIEWER.snapshot()'), observations);assert.equal(v.timers.length, 0);}
  function checkROC(m, n, h, mode) {
    v.execute(`MIA_FIXED_INTRO.setROC(${JSON.stringify({members:m,high:n,hits:h,mode})})`);
    const model = v.json('MIA_FIXED_INTRO.rocModel()'), expected = independentROC(m, n, h);
    close(model.score, expected.score); assert.deepEqual(model.points[0], [0,0]);
    close(model.points[1][0], expected.points[1][0]);close(model.points[1][1], expected.points[1][1]);
    assert.equal(model.highMembers + model.lowMembers, m); assert.equal(model.highNonmembers + model.lowNonmembers, 24-m);
    assert.equal(model.highMembers + model.highNonmembers, n);
    for (const id of ['roc-full', 'roc-zoom']) {
      const markup = v.elements.get(id).innerHTML;
      assert(!/NaN|Infinity/.test(markup));
      const point = markup.match(/data-fpr="([^"]+)" data-tpr="([^"]+)" cx="([^"]+)" cy="([^"]+)"/);assert(point);
      close(Number(point[1]), .01);close(Number(point[2]), expected.score);assert(Number(point[3])>=56&&Number(point[3])<=386);assert(Number(point[4])>=30&&Number(point[4])<=220);
    }
    const dots = (id, label) => [...v.elements.get(id).innerHTML.matchAll(new RegExp(`aria-label="${label}"`, 'g'))].length;
    assert.equal(dots('roc-high-people', '所属者'), h);assert.equal(dots('roc-high-people', '非所属者'), n-h);
    assert.equal(dots('roc-low-people', '所属者'), m-h);assert.equal(dots('roc-low-people', '非所属者'), 24-m-n+h);
    assert.equal(v.elements.get('roc-hits').disabled, mode === 'population');unchanged();report.roc_configurations++;
  }
  for (const m of [1,2]) for(let n=11;n<=24;n++) checkROC(m,n,m,'population');
  for (const m of [1,2]) for(let n=6;n<=12;n++) for(let h=0;h<=m;h++)checkROC(m,n,h,'subset');
  v.select('roc-mode','population');assert.equal(v.json('MIA_FIXED_INTRO.rocModel()').high,11);v.select('roc-members',2);assert.equal(v.json('MIA_FIXED_INTRO.rocModel()').hits,2);
  v.elements.get('roc-high').value='24';v.elements.get('roc-high').oninput();close(v.json('MIA_FIXED_INTRO.rocModel()').score,.01);
  v.select('roc-mode','subset');v.select('roc-hits',0);close(v.json('MIA_FIXED_INTRO.rocModel()').score,0);v.select('roc-members',1);assert.equal(v.elements.get('roc-hits').value,'0');
  v.click('roc-show-zoom');assert.equal(v.elements.get('roc-plots').attributes['data-plot'],'zoom');v.click('roc-show-full');assert.equal(v.elements.get('roc-plots').attributes['data-plot'],'full');unchanged();
  assert.throws(()=>v.execute('MIA_FIXED_INTRO.setROC({members:0})'),/人数設定/);
  assert.throws(()=>v.execute('MIA_FIXED_INTRO.setROC({high:25})'),/人数設定/);
  assert.throws(()=>v.execute('MIA_FIXED_INTRO.setROC({hits:2})'),/人数設定/);
  v.click('intro-next');assert.equal(v.json('MIA_FIXED_INTRO.snapshot()').page,'history');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').stage,0);
  assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').remaining,null);assert.deepEqual(v.json('MIA_FIXED_INTRO.historyModel()').observed,[null,null]);assert.equal(v.elements.get('history-case-controls').hidden,true);
  for(const team of c.teams){v.select('history-team',team);const i=c.teams.indexOf(team), markup=v.elements.get('history-setting').innerHTML;assert(markup.includes(`高側${c.round1_sizes[i]}人`));assert(markup.includes(`高側${c.round2_sizes[i]}人`));report.historical_team_settings++;unchanged();}
  v.click('intro-next');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').remaining,102);
  const predictions = [];
  for(const row of source.cases){
    v.select('history-case',row.case_id);const model=v.json('MIA_FIXED_INTRO.historyModel()');
    close(model.predicted[0],independentMean(row,1));close(model.predicted[0],c.round1_rare_score);close(model.predicted[1],independentMean(row,2));
    const [num,den]=row.round2_score_exact.split('/').map(Number);close(model.predicted[1],num/den);
    assert.deepEqual(model.counts,row.counts);assert.equal(model.activeTeams,Object.values(row.counts).filter(x=>x).length);assert.equal(model.totalMembers,Object.values(row.counts).reduce((a,b)=>a+b,0));
    const table=v.elements.get('history-team-grid').innerHTML;assert.equal([...table.matchAll(/class="history-team-cell/g)].length,22);
    const dots=[...v.elements.get('history-case-plot').innerHTML.matchAll(/data-history-case="([^"]+)" data-predicted-score="([^"]+)"/g)];assert.equal(dots.length,102);
    dots.forEach(dot=>close(Number(dot[2]),source.cases.find(x=>x.case_id===dot[1]).round2_score));
    predictions.push(model.predicted[1]);unchanged();report.historical_configurations++;
  }
  assert.equal(new Set(source.cases.map(row=>row.round2_score_exact)).size,102);
  assert.deepEqual(source.cases.filter((row,i)=>Math.abs(predictions[i]-c.round2_rare_score)<1e-13).map(row=>row.case_id),['N015']);
  v.click('intro-next');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').remaining,1);assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').caseID,'N015');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').matchesSecond,true);
  assert.deepEqual(v.json('MIA_FIXED_INTRO.historyModel()').observed,[c.round1_rare_score,c.round2_rare_score]);
  const markup=v.elements.get('history-case-plot').innerHTML;assert.equal([...markup.matchAll(/history-dot eliminated/g)].length,101);assert(markup.includes('第2回の実値に一致：N015'));
  v.select('history-case','N001');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').matchesSecond,false);assert(v.elements.get('history-selected-caption').textContent.includes('残らない'));
  v.elements.get('history-case-plot').onclick({target:{closest(){return {dataset:{historyCase:'N015'}};}}});assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').matchesSecond,true);
  v.click('intro-next');assert.equal(v.json('MIA_FIXED_INTRO.snapshot()').page,'bridge');
  const bridge=v.json('MIA_FIXED_INTRO.bridgeModel()'), high=[6,7,7,11,10,12], caps=[1,2,1,2,1,2];
  high.forEach((n,i)=>bridge.individualScores[i].forEach((score,h)=>{close(score,independentROC(caps[i],n,h).score);report.bridge_score_cases++;}));
  close(bridge.mean,Number(v.execute('MIA_FIXED_VIEWER.meanExact([0,1,0,1,1,1]).split("/")[0]'))/6750);
  assert.equal(v.execute('MIA_FIXED_VIEWER.meanExact([0,1,0,1,1,1])'),'73/6750');
  const mean=bridge.responses.reduce((sum,h,i)=>sum+independentROC(caps[i],high[i],h).score,0)/6;close(mean,bridge.mean);
  assert(html.includes('216通り'));assert(html.includes('実提出の第3〜7回の符号とは異なります'));unchanged();
  v.click('intro-next');assert.equal(v.json('MIA_FIXED_OVERVIEW.snapshot()').view,'assignment');assert.equal(v.json('MIA_FIXED_ASSIGNMENT.snapshot()').placed.length,0);unchanged();
  v.click('show-intro');v.click('intro-prev');assert.equal(v.json('MIA_FIXED_INTRO.snapshot()').page,'history');v.click('intro-prev');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').stage,1);v.click('intro-prev');v.click('intro-prev');assert.equal(v.json('MIA_FIXED_INTRO.snapshot()').page,'roc');unchanged();
  v.click('show-overview');const run=v.click('animate-all');v.click('show-intro');const paused=v.json('MIA_FIXED_LESSON.snapshot()');assert.equal(paused.playing,false);assert.equal(paused.phase.stage,'result');await v.tick();await run;assert.deepEqual(v.json('MIA_FIXED_LESSON.snapshot()'),paused);
  if(process.argv.includes('--write-report'))fs.writeFileSync(path.join(__dirname,'validation_fixed_intro.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
