'use strict';
const assert=require('node:assert/strict'), fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {simulatedViewer}=require('./test_fixed_ui.cjs');
const root=path.resolve(__dirname,'../..'),html=fs.readFileSync(path.join(root,'docs/visualizer/MIA_5query_cards.html'),'utf8');
const data=JSON.parse(html.match(/<script id="fixed-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const close=(a,b)=>assert(Math.abs(a-b)<1e-13,`${a} != ${b}`);
const numeric=ratio=>{const [n,d]=ratio.split('/').map(Number);return n/d;};
// Independent scorer: count labels at tied confidence thresholds and interpolate at FPR .01.
function roc(m,n,h){
  const rows=[...Array.from({length:h},()=>[.4,1]),...Array.from({length:n-h},()=>[.4,0]),...Array.from({length:m-h},()=>[.1,1]),...Array.from({length:24-m-n+h},()=>[.1,0])].sort((a,b)=>b[0]-a[0]);
  const points=[[0,0]];let fp=0,tp=0;
  rows.forEach((row,i)=>{tp+=row[1];fp+=1-row[1];if(i===rows.length-1||row[0]!==rows[i+1][0])points.push([fp/(24-m),tp/m]);});
  const i=points.findIndex(p=>p[0]>=.01),a=points[i-1],b=points[i];return a[1]+(.01-a[0])/(b[0]-a[0])*(b[1]-a[1]);
}
function main(){
  const v=simulatedViewer(),lesson=v.json('MIA_FIXED_LESSON.snapshot()'),operational=v.json('MIA_FIXED_VIEWER.snapshot()');
  const report={status:'passed',environment:'Node simulated DOM; independent 24-record tied-score ROC; exact saved mean encoding; all finite hypotheses',finite_mean_combinations:0,historical_expanded_means:0,zero_tpr_included_in_six_team_mean:true,manual_controls_have_no_timers:true,visual_browser_validation:false,html_bytes:Buffer.byteLength(html),html_sha256:crypto.createHash('sha256').update(html).digest('hex')};
  const unchanged=()=>{assert.equal(v.timers.length,0);assert.deepEqual(v.json('MIA_FIXED_LESSON.snapshot()'),lesson);assert.deepEqual(v.json('MIA_FIXED_VIEWER.snapshot()'),operational);};
  v.click('intro-tab-bridge');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,0);assert(v.elements.get('bridge-compare').disabled);assert(v.elements.get('bridge-compare-all').disabled);
  v.click('bridge-compare');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,0);
  v.click('bridge-calculate');let state=v.json('MIA_FIXED_MEAN.model()');assert.equal(state.stage,1);assert.equal(state.contributions.length,6);assert.equal(state.meanExact,'0/1');assert(v.elements.get('bridge-example').innerHTML.includes('6チーム'));
  v.click('bridge-compare');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,2);assert(v.elements.get('bridge-comparison').innerHTML.includes('不一致'));unchanged();
  const all=[];
  const check=hits=>{
    v.execute(`MIA_FIXED_MEAN.setHits(${JSON.stringify(hits)})`);state=v.json('MIA_FIXED_MEAN.model()');assert.equal(state.stage,0);assert.equal(v.elements.get('bridge-comparison').hidden,true);assert.equal(v.elements.get('bridge-inferred').hidden,true);
    const scores=hits.map((h,i)=>roc(data.caps[i],data.high[i],h));
    state.contributions.forEach((x,i)=>{assert.equal(x.team,data.teams[i]);assert.equal(x.hits,hits[i]);close(x.value,scores[i]);close(numeric(x.exact),scores[i]);assert.equal(v.elements.get(`bridge-hits-${i}`).value,String(hits[i]));assert(v.elements.get(`bridge-tpr-${i}`).innerHTML.includes(x.exact));});
    close(numeric(state.sumExact),scores.reduce((a,b)=>a+b,0));close(state.meanValue,scores.reduce((a,b)=>a+b,0)/6);
    assert.equal(state.meanExact,v.execute(`MIA_FIXED_VIEWER.meanExact(${JSON.stringify(hits)})`));
    v.click('bridge-calculate');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,1);v.click('bridge-compare');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,2);
    assert.equal(state.matchesTarget,hits.join('')==='010111');assert(v.elements.get('bridge-comparison').innerHTML.includes(state.matchesTarget?'この組は一致':'この組は不一致'));all.push(state);report.finite_mean_combinations++;unchanged();
  };
  function enumerate(prefix){if(prefix.length===6){check(prefix);return;}for(let h=0;h<=data.caps[prefix.length];h++)enumerate([...prefix,h]);}enumerate([]);
  assert.equal(all.length,216);assert.equal(new Set(all.map(x=>x.meanExact)).size,216);assert.deepEqual(all.filter(x=>x.matchesTarget).map(x=>x.hits),[[0,1,0,1,1,1]]);
  v.click('bridge-compare-all');state=v.json('MIA_FIXED_MEAN.model()');assert.equal(state.stage,3);assert.equal(state.matchCount,1);assert.deepEqual(state.inferred,[0,1,0,1,1,1]);assert.equal(state.meanExact,'73/6750');assert.equal(v.elements.get('bridge-inferred').hidden,false);
  const plot=v.elements.get('bridge-comparison').innerHTML, dots=[...plot.matchAll(/data-mean-hits="([012]{6})" data-mean-exact="([^"]+)"/g)];assert.equal(dots.length,216);assert.equal([...plot.matchAll(/mean-candidate excluded/g)].length,215);assert.equal([...plot.matchAll(/mean-candidate matched/g)].length,1);
  dots.forEach(dot=>assert.equal(dot[2],all.find(x=>x.hits.join('')===dot[1]).meanExact));unchanged();
  const coordinates=[...plot.matchAll(/cx="([^"]+)" cy="([^"]+)"/g)];assert.equal(coordinates.length,216);coordinates.forEach(p=>{assert(Number(p[1])>=28&&Number(p[1])<=652);assert(Number(p[2])>=18&&Number(p[2])<=36);});
  v.select('bridge-hits-1',2);assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,0);assert.equal(v.json('MIA_FIXED_MEAN.model()').hits[1],2);
  v.click('bridge-reset');assert.deepEqual(v.json('MIA_FIXED_MEAN.model()').hits,[0,0,0,0,0,0]);assert.throws(()=>v.execute('MIA_FIXED_MEAN.setHits([2,0,0,0,0,0])'),/高側人数/);unchanged();
  v.click('intro-tab-history');v.click('history-stage-1');assert.equal(v.elements.get('history-mean-panel').hidden,false);assert.equal(v.elements.get('history-team-grid').hidden,true);
  const firstMeans=new Set(),secondMeans=new Set();
  for(const row of data.intro.cases){
    v.select('history-case',row.case_id);
    for(const round of [1,2]){
      v.click(`history-mean-round-${round}`);const model=v.json('MIA_FIXED_INTRO.historyModel()'),details=model.meanDetails,c=data.intro.conditions,sizes=c[round===1?'round1_sizes':'round2_sizes'];
      const active=c.teams.filter(t=>row.counts[t]>0);assert.equal(details.denominator,active.length);assert.equal(details.excluded,22-active.length);assert.deepEqual(details.contributions.map(x=>x.team),active);
      let total=0;details.contributions.forEach(x=>{const n=sizes[c.teams.indexOf(x.team)],m=row.counts[x.team],expected=roc(m,n,m);assert.equal(x.high,n);assert.equal(x.members,m);close(numeric(x.exact),expected);total+=expected;});
      close(numeric(details.sumExact),total);close(numeric(details.meanExact),total/active.length);close(details.meanValue,total/active.length);
      const markup=v.elements.get('history-mean-contributions').innerHTML;assert.equal([...markup.matchAll(/data-mean-team=/g)].length,active.length);const equation=v.elements.get('history-mean-equation').innerHTML;assert(equation.includes(`${active.length}チーム`));assert(equation.includes(details.meanExact));
      if(round===1){close(details.meanValue,c.round1_rare_score);firstMeans.add(details.meanExact);assert(equation.includes('一致'));}else{assert.equal(details.meanExact,row.round2_score_exact);secondMeans.add(details.meanExact);assert(equation.includes('まだ表示していません'));}
      report.historical_expanded_means++;unchanged();
    }
  }
  assert.equal(firstMeans.size,1);assert.equal(secondMeans.size,102);assert(data.intro.cases.some(row=>row.active_teams===7));
  v.click('history-show-configurations');assert.equal(v.elements.get('history-mean-panel').hidden,true);assert.equal(v.elements.get('history-team-grid').hidden,false);assert.equal(v.elements.get('history-case-plot').hidden,false);v.click('history-show-mean');assert.equal(v.elements.get('history-mean-panel').hidden,false);
  v.click('history-stage-2');assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').meanRound,2);assert.equal(v.json('MIA_FIXED_INTRO.historyModel()').meanDetails.denominator,6);assert(v.elements.get('history-mean-equation').innerHTML.includes('一致'));
  v.select('history-case',data.intro.cases.find(row=>row.case_id!==data.intro.winner).case_id);assert(v.elements.get('history-mean-equation').innerHTML.includes('不一致'));unchanged();
  v.click('intro-tab-bridge');v.click('bridge-reset');for(let s=1;s<=3;s++){v.click('intro-next');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,s);unchanged();}
  v.click('intro-prev');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,2);v.click('intro-next');assert.equal(v.json('MIA_FIXED_MEAN.model()').stage,3);v.click('intro-next');assert.equal(v.json('MIA_FIXED_OVERVIEW.snapshot()').view,'assignment');unchanged();
  assert(html.includes('平均1つから自由な6つのTPRを求めることはできません'));assert(html.includes('TPR＝0も平均に含めます'));
  if(process.argv.includes('--write-report'))fs.writeFileSync(path.join(__dirname,'validation_fixed_mean.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}
main();
