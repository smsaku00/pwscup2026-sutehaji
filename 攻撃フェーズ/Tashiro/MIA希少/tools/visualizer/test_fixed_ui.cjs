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
const report = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/resources/size_report_5query.json'), 'utf8'));
const sourceCode = JSON.parse(fs.readFileSync(path.join(root, 'docs/visualizer/resources/nonadaptive_code.json'), 'utf8'));
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


function simulatedViewer(reduced = false) {
  const elements = new Map(), downloads = [], blobs = new Map(), timers = [], events = {};
  class Element {
    constructor(tag = 'div', id = '', attributes = {}) {
      this.tagName = tag.toUpperCase(); this.id = id; this.value = ''; this.checked = false;
      this.hidden = false; this.disabled = false; this.textContent = ''; this.children = [];
      this.dataset = {}; this.attributes = {...attributes}; this._html = ''; this.clientHeight = 420;
      this.style = {setProperty(key,value) {this[key] = value;}};
      const classes = new Set((attributes.class || '').split(/\s+/));
      this.classList = {toggle(key,on) {if(on) classes.add(key); else classes.delete(key);}, remove(key) {classes.delete(key);}, contains(key) {return classes.has(key);}};
    }
    setAttribute(key,value) {this.attributes[key]=String(value);if(key.startsWith('data-')) this.dataset[key.slice(5)] = String(value);}
    set innerHTML(value) {
      this._html=String(value); this.children=[];
      const options=[...this._html.matchAll(/<option\b([^>]*)\bvalue="([^"]+)"([^>]*)>/g)];
      if(this.tagName==='SELECT' && options.length) this.value=(options.find(m => (m[1]+m[3]).includes('selected')) || options[0])[2];
    }
    get innerHTML() {return this._html;}
    replaceChildren(...children) {this.children=children;this._html='';}
    append(...children) {this.children.push(...children);}
    querySelectorAll() {return [];}
    focus() {document.activeElement=this;}
    click() {if(this.disabled) return;if(this.tagName==='A') {downloads.push({name:this.download,blob:blobs.get(this.href)});return;} return this.onclick?.();}
  }
  for(const m of html.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)) {
    const element=new Element(m[1],m[3],attrs(m[2])); element.hidden=/\bhidden\b/.test(m[2]); elements.set(m[3],element);
  }
  elements.get(attrs(dataScript[1]).id).textContent=dataScript[2];
  const document={activeElement:null,getElementById(id) {const el=elements.get(id);assert(el,`Unknown DOM id ${id}`);return el;},createElement(tag) {return new Element(tag);},addEventListener(name,callback) {events[name]=callback;}};
  const location={hash:'',pathname:'/MIA_5query_cards.html',search:''};
  const sandbox={document,location,history:{replaceState() {}},Blob,URL:{createObjectURL(blob) {const key=`blob:test-${blobs.size}`;blobs.set(key,blob);return key;},revokeObjectURL() {}},
    setTimeout(callback,ms) {timers.push({callback,ms});return timers.length;},requestAnimationFrame(callback) {queueMicrotask(callback);},addEventListener(name,callback) {events[name]=callback;},matchMedia() {return {matches:reduced};},console};
  sandbox.window=sandbox;
  const context=vm.createContext(sandbox);
  scripts.filter(script=>script!==dataScript).forEach((script,i)=>vm.runInContext(script[2],context,{filename:`fixed-page-script-${i}.js`}));
  const execute=code=>vm.runInContext(code,context);
  const json=code=>JSON.parse(execute(`JSON.stringify(${code})`));
  const select=(id,value)=>{const el=elements.get(id);el.value=String(value);assert(el.onchange);el.onchange();};
  const click=id=>elements.get(id).click();
  const tick=async()=>{assert(timers.length);timers.shift().callback();await Promise.resolve();await Promise.resolve();await Promise.resolve();await Promise.resolve();};
  return {elements,execute,json,select,click,downloads,timers,tick,events,document};
}

const pairPaths=markup=>[...markup.matchAll(/<path class="pair-line([^\"]*)"([^>]*)>/g)].map(m=>({classes:m[1],attrs:attrs(m[2])}));
const recordRows=markup=>[...markup.matchAll(/<div class="pair-record([^\"]*)"([^>]*)>([\s\S]*?)<\/div>/g)].map(m=>({classes:m[1],attrs:attrs(m[2]),body:m[3]}));
function assertOverview(viewer,result) {
  const state=viewer.json('MIA_FIXED_OVERVIEW.snapshot()'), lesson=viewer.json('MIA_FIXED_LESSON.snapshot()');
  assert.equal(viewer.elements.get('lesson-workspace').attributes['data-view'],state.view);
  assert.equal(viewer.elements.get('overview-panel').hidden,!['overview','both'].includes(state.view));
  assert.equal(viewer.elements.get('pairs-panel').hidden,!['cards','both'].includes(state.view));
  assert.equal(viewer.elements.get('assignment-panel').hidden,state.view!=='assignment');
  assert.equal(viewer.elements.get('intro-panel').hidden,state.view!=='intro');
  assert.equal(viewer.elements.get('lesson-controls').hidden,['assignment','intro'].includes(state.view));
  assert.equal(viewer.elements.get('lesson-actions').hidden,['assignment','intro'].includes(state.view));
  for(const [id,view] of [['show-intro','intro'],['show-assignment','assignment'],['show-overview','overview'],['show-cards','cards'],['show-both','both']])assert.equal(viewer.elements.get(id).attributes['aria-pressed'],String(state.view===view));
  let used=0;state.memberships.forEach((member,t)=>{assert.equal(pc(member),CAPS[t]);assert.equal(used & member,0);used|=member;});assert.equal(pc(used),9);
  assert.equal(CAPS[state.selectedTeam],lesson.cap);assert.equal(state.memberships[state.selectedTeam],lesson.member);
  assert.equal(state.depth,lesson.depth);assert.equal(state.revealedCount,lesson.revealedCount);
  const observed=QUERIES.map((query,r)=>r<lesson.revealedCount?state.memberships.map(member=>pc(member & query)):null);
  const columns=CAPS.map((_,t)=>observed.map(row=>row===null?null:row[t]));
  assert.deepEqual(state.responses,observed);assert.deepEqual(state.columns,columns);
  assert.deepEqual(state.rowSequences,observed.map(row=>row===null?'······':row.join('')));
  assert.deepEqual(state.meanFractions,observed.map(row=>row===null?null:exactMean(row)));
  const known=observed.map((row,r)=>r<lesson.depth?row:null), expectedSets=independentSets(known);
  state.candidateSets.forEach((set,t)=>assert.deepEqual(sorted(set),expectedSets[t]));assert.deepEqual(state.counts,expectedSets.map(set=>set.length));
  const markup=viewer.elements.get('overview-table').innerHTML;
  const values=[...markup.matchAll(/<td data-overview-cell-team="(\d+)" data-overview-cell-round="(\d+)" data-pair-selectable="([01])" data-hit="([012]?)"[^>]*>([012·])<\/td>/g)];
  assert.equal(values.length,30);
  values.forEach(m=>{const t=Number(m[1]),r=Number(m[2]);assert.equal(m[3],'1');assert.equal(m[4],observed[r]===null?'':String(observed[r][t]));assert.equal(m[5],observed[r]===null?'·':String(observed[r][t]));result.overview_cell_checks++;});
  const columnMarkup=[...markup.matchAll(/<td data-overview-column="(\d+)"[^>]*>([012·]{5})<\/td>/g)];assert.equal(columnMarkup.length,6);
  columnMarkup.forEach(m=>assert.equal(m[2],columns[Number(m[1])].map(n=>n===null?'·':n).join('')));
  assert(!markup.includes('data-overview-member='));
  const buttons=[...markup.matchAll(/data-overview-team="(\d+)"/g)].map(m=>Number(m[1]));assert.deepEqual(buttons,[0,1,2,3,4,5]);
  assert.equal(viewer.elements.get('lesson-team').value,String(data.teams[state.selectedTeam]));
  for(let i=0;i<5;i++) assert.equal(viewer.elements.get(`record-heading-${i+1}`).classList.contains('row-focus'),i===state.selectedRound);
  result.overview_states_checked++;
}
function assertGraph(viewer,member,depth,stage,result) {
  const capacity=pc(member), rows=independentTrace(capacity,member), before=depth?rows[depth-1].after:enumerate(capacity), after=stage==='removing'?rows[depth].after:before;
  const expectedRemoved=stage==='removing'?before.filter(mask=>!after.includes(mask)):[];
  const state=viewer.json('MIA_FIXED_LESSON.snapshot()');assert.equal(state.cap,capacity);assert.equal(state.member,member);assert.equal(state.depth,depth);
  assert.equal(state.phase?.stage||null,stage);assert.deepEqual(sorted(state.remainingMasks),before);assert.deepEqual(sorted(state.removedMasks),expectedRemoved);
  const revealed=depth+(stage?1:0);assert.equal(state.revealedCount,revealed);assert.deepEqual(state.results,signature(member).map((n,r)=>r<revealed?n:null));
  const paths=pairPaths(viewer.elements.get('pair-lines').innerHTML);assert.deepEqual(sorted(paths.map(p=>Number(p.attrs['data-pair-mask']))),capacity===2?before:[]);
  paths.forEach(p=>{
    const mask=Number(p.attrs['data-pair-mask']), fading=expectedRemoved.includes(mask);
    assert.equal(p.attrs['data-fading'],fading?'1':'0');assert.equal(p.classes.includes('fading'),fading);
    assert.equal(p.attrs['tabindex'],fading?'-1':'0');
    assert.equal(p.attrs['data-current-sum'],revealed?String(pc(mask & QUERIES[revealed-1])):'');
    assert.equal(p.classes.includes('unique'),before.length===1 && !stage);result.edge_checks++;
  });
  const active=IDS.filter((_,i)=>before.some(mask=>mask>>i&1)), next=IDS.filter((_,i)=>after.some(mask=>mask>>i&1));
  assert.deepEqual(state.activeIDs,active);assert.deepEqual(state.nextActiveIDs,next);
  const records=recordRows(viewer.elements.get('record-deck').innerHTML);assert.equal(records.length,11);
  records.forEach((record,i)=>{
    assert.equal(Number(record.attrs['data-record-id']),IDS[i]);const alive=active.includes(IDS[i]), fading=alive&&!next.includes(IDS[i]);
    assert.equal(record.classes.includes('excluded'),!alive);assert(!record.classes.includes('gone'));assert.equal(record.attrs['role'],'group');assert.equal(record.attrs['aria-label'],`ID ${IDS[i]}：${!alive?'候補外':fading?'候補外になります':'候補'}`);assert(!('aria-hidden' in record.attrs));assert.equal(record.attrs['data-active'],alive?'1':'0');assert.equal(record.attrs['data-fading'],fading?'1':'0');
    const bits=[...record.body.matchAll(/data-round="(\d+)" data-value="([01])"/g)];assert.equal(bits.length,5);
    bits.forEach((m,r)=>{assert.equal(Number(m[1]),r);assert.equal(Number(m[2]),(QUERIES[r]>>i)&1);result.record_bit_checks++;});
  });
  const chips=[...viewer.elements.get('result-strip').innerHTML.matchAll(/data-round="(\d+)" data-result="([012]?)"[^>]*><span>提出\d<\/span><b>([012—])<small>/g)];assert.equal(chips.length,5);
  chips.forEach((m,r)=>{assert.equal(Number(m[1]),r);assert.equal(m[2],r<revealed?String(signature(member)[r]):'');assert.equal(m[3],r<revealed?String(signature(member)[r]):'—');});
  const counts=[enumerate(capacity).length,...rows.slice(0,depth).map(row=>row.after.length)];assert.equal(viewer.elements.get('count-path').textContent,counts.join(' → '));
  assert(!viewer.elements.get('filter-rule').textContent.includes('答え'));
  if(depth===5){assert.deepEqual(before,[member]);assert.equal(active.length,capacity);assert(viewer.elements.get('summary-result').textContent.includes(`ID ${label(member).replaceAll('・','＋')}`));}
  assertOverview(viewer,result);result.graph_states_checked++;
}
async function main() {
  const result={status:'passed',environment:'Node VM simulated DOM, independently enumerated one/two-member sets, controlled reveal/fade timers and numeric SVG geometry',
    html_bytes:Buffer.byteLength(html),html_sha256:crypto.createHash('sha256').update(html).digest('hex'),single_member_examples:0,two_member_examples:0,graph_states_checked:0,edge_checks:0,record_bit_checks:0,overview_states_checked:0,overview_cell_checks:0,animated_submission_cases:0,manual_submission_cases:0,
    controls_checked:[],formulas_checked:[],mean_decodes_checked:[],negative_cases:[],visual_browser_validation:false};
  const viewer=simulatedViewer();assert.deepEqual(viewer.json('MIA_FIXED_LESSON.values()'),{cap:2,depth:0,masks:{'1':128,'2':129},playing:false});
  assert.deepEqual(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships,[2,129,4,24,32,320]);assertGraph(viewer,129,0,null,result);
  assert.equal(pairPaths(viewer.elements.get('pair-lines').innerHTML).length,55);
  assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').view,'intro');assert.equal(viewer.elements.get('intro-panel').hidden,false);assert.equal(viewer.elements.get('overview-panel').hidden,true);assert.equal(viewer.elements.get('pairs-panel').hidden,true);viewer.click('show-assignment');
  result.assignment_checks={records:0,cells:0,manual_steps:0,automatic_steps:0};
  function checkAssignment(v) {
    const state=v.json('MIA_FIXED_ASSIGNMENT.snapshot()'), a=result.assignment_checks;
    assert.equal(state.selectedID,IDS[state.selected]);assert.deepEqual(state.pattern,QUERIES.map(query=>query>>state.selected&1));
    assert.equal(new Set(state.placed).size,state.placed.length);
    const expected=QUERIES.map(query=>IDS.filter((_,i)=>state.placed.includes(i)&&(query>>i&1)));
    assert.deepEqual(state.destinations,expected);
    const grid=v.elements.get('assignment-grid').innerHTML;
    assert.equal([...grid.matchAll(/data-assignment-row="/g)].length,11);
    const cells=[...grid.matchAll(/data-assignment-cell="(\d+):(\d+)" data-bit="([01])"/g)];assert.equal(cells.length,55);
    cells.forEach(m=>{assert.equal(Number(m[3]),QUERIES[Number(m[2])]>>Number(m[1])&1);a.cells++;});
    const decisions=[...v.elements.get('assignment-focus').innerHTML.matchAll(/data-assignment-decision="(\d+)" data-bit="([01])"/g)];assert.equal(decisions.length,5);
    decisions.forEach(m=>assert.equal(Number(m[2]),state.pattern[Number(m[1])]));
    const dest=v.elements.get('assignment-destinations').innerHTML;
    const columns=[...dest.matchAll(/data-assignment-destination="(\d+)"([\s\S]*?)(?=<div class="assignment-destination[ "]|$)/g)];assert.equal(columns.length,5);
    columns.forEach(m=>assert.deepEqual([...m[2].matchAll(/data-destination-id="(\d+)"/g)].map(n=>Number(n[1])),expected[Number(m[1])]));
    assert.equal(v.elements.get('assignment-step').disabled,state.playing||state.placed.length===11);
    assert.equal(v.elements.get('assignment-all').disabled,state.playing||state.placed.length===11);
    assert.equal(v.elements.get('assignment-record').value,String(state.selectedID));
    return state;
  }
  checkAssignment(viewer);
  const unchangedLesson=viewer.json('MIA_FIXED_LESSON.snapshot()');
  for(let i=0;i<11;i++) {
    viewer.select('assignment-record',IDS[i]);checkAssignment(viewer);result.assignment_checks.records++;
    viewer.click('assignment-step');assert.equal(viewer.timers.length,0);const placed=checkAssignment(viewer);
    assert.deepEqual(placed.placed,Array.from({length:i+1},(_,j)=>j));result.assignment_checks.manual_steps++;
    if(i<10) {viewer.click('assignment-step');assert.equal(checkAssignment(viewer).selected,i+1);assert.equal(viewer.timers.length,0);}
    assert.deepEqual(viewer.json('MIA_FIXED_LESSON.snapshot()'),unchangedLesson);
  }
  assert.deepEqual(checkAssignment(viewer).destinations,[[76,169,264,462,675],[47,90,115,264,675],[76,90,264,383,495],[47,76,90,169,304],[47,76,115,462,495]]);
  viewer.click('assignment-reset');assert.equal(checkAssignment(viewer).placed.length,0);
  viewer.elements.get('assignment-grid').onclick({target:{closest(){return {dataset:{assignmentRecord:'7'}};}}});assert.equal(checkAssignment(viewer).selectedID,383);
  viewer.click('assignment-all');assert.equal(checkAssignment(viewer).placed.length,11);assert.equal(viewer.timers.length,0);
  viewer.click('assignment-reset');const assignmentAuto=viewer.click('assignment-play');
  for(let i=0;i<11;i++) {
    assert.equal(checkAssignment(viewer).selected,i);assert.equal(viewer.timers[0].ms,1600);
    await viewer.tick();const state=checkAssignment(viewer);assert.equal(state.placed.length,i+1);assert.equal(state.moving,i);assert.equal(viewer.timers[0].ms,1800);
    await viewer.tick();result.assignment_checks.automatic_steps++;
  }
  await assignmentAuto;assert.equal(checkAssignment(viewer).playing,false);assert.equal(viewer.timers.length,0);
  viewer.click('assignment-reset');const pausedAssignment=viewer.click('assignment-play');viewer.click('assignment-play');const paused=checkAssignment(viewer);
  await viewer.tick();await pausedAssignment;assert.deepEqual(checkAssignment(viewer),paused);assert.equal(paused.placed.length,0);
  viewer.click('assignment-step');assert.equal(checkAssignment(viewer).placed.length,1);assert.equal(viewer.timers.length,0);
  const leavingAssignment=viewer.click('assignment-play');await viewer.tick();viewer.click('assignment-next');const afterLeaving=checkAssignment(viewer);
  assert.equal(afterLeaving.playing,false);assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').view,'overview');
  await viewer.tick();await leavingAssignment;assert.deepEqual(checkAssignment(viewer),afterLeaving);
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.snapshot()'),unchangedLesson);
  viewer.click('show-assignment');assertOverview(viewer,result);viewer.click('show-overview');
  assert.throws(()=>viewer.execute('MIA_FIXED_ASSIGNMENT.select(11)'),/レコード/);
  const resetAssignment=viewer.click('assignment-play');viewer.click('assignment-reset');await viewer.tick();await resetAssignment;assert.equal(checkAssignment(viewer).placed.length,0);
  const reducedAssignment=simulatedViewer(true), reducedAssignmentRun=reducedAssignment.click('assignment-play');await reducedAssignment.tick();assert.equal(reducedAssignment.timers[0].ms,1600);reducedAssignment.click('assignment-play');await reducedAssignment.tick();await reducedAssignmentRun;
  const switchViewer=simulatedViewer();switchViewer.click('show-overview');const hiddenAuto=switchViewer.click('animate-all');
  switchViewer.click('show-assignment');const pausedResult=switchViewer.json('MIA_FIXED_LESSON.snapshot()');
  assert.equal(pausedResult.playing,false);assert.equal(pausedResult.phase.stage,'result');assert.equal(pausedResult.depth,0);assert.equal(pausedResult.revealedCount,1);
  await switchViewer.tick();await hiddenAuto;assert.deepEqual(switchViewer.json('MIA_FIXED_LESSON.snapshot()'),pausedResult);
  switchViewer.click('show-cards');const applyAfterPrep=switchViewer.click('merge-one');await switchViewer.tick();await applyAfterPrep;assert.equal(switchViewer.json('MIA_FIXED_LESSON.snapshot()').depth,1);assert.equal(switchViewer.timers.length,0);
  const selectionViewer=simulatedViewer(), changingAssignment=selectionViewer.click('assignment-play');await selectionViewer.tick();selectionViewer.select('assignment-record',495);const selectedWhileMoving=checkAssignment(selectionViewer);await selectionViewer.tick();await changingAssignment;assert.deepEqual(checkAssignment(selectionViewer),selectedWhileMoving);
  result.controls_checked.push('11 horizontal five-bit patterns independently match all 55 cells and the five vertical submission sets', 'manual assignment and next-record clicks have no timer; only explicit record playback is automatic', 'assignment pause/reset/selection/view changes invalidate old timers without changing results or memberships', 'returning to assignment pauses result playback and preserves its pending result for manual filtering');
  assert(viewer.elements.has('tab-single'));assert(viewer.elements.has('tab-pair'));assert(!viewer.elements.has('target-row'));
  assert.equal(viewer.elements.get('playback-speed').value,'slow');assert.deepEqual(viewer.json('MIA_FIXED_LESSON.timing()'),{result:1600,fade:1700,gap:1000});
  assert.equal(viewer.elements.get('pair-stage').style['--fade-ms'],'1700ms');
  assert.equal(viewer.elements.get('answer-record-2-wrap').hidden,false);
  const staticMarkup=html.split('<script>')[0];
  for(const id of ['tab-single','tab-pair','lesson-team','answer-record-1','answer-record-2','playback-speed']) {
    assert.equal([...staticMarkup.matchAll(new RegExp('id="'+id+'"','g'))].length,1);
    assert(staticMarkup.indexOf('id="'+id+'"')<staticMarkup.indexOf('id="help-overlay"'));
  }assert(!staticMarkup.includes('答え'));assert(!staticMarkup.includes('質問'));assert(!staticMarkup.includes('調査'));
  const oldPatterns=Buffer.from(fs.readFileSync(path.join(root,'docs/visualizer/archive/MIA_5query_patterns.html'),'utf8').replace(/href="(?:\.\.\/)?(?:resources\/)?(MIA_[^"]+\.html|nonadaptive_code\.json)"/g,'href="$1"'));assert.equal(oldPatterns.length,80630);
  assert.equal(crypto.createHash('sha256').update(oldPatterns).digest('hex'),'393f21626465099138e3780d2dff546fab0536e6321cc8bba08533864e80cb2c');
  result.team_answer_cases=0;
  for(const team of [0,1,2,3,4,5]) {
    viewer.execute(`MIA_FIXED_OVERVIEW.selectTeam(${team})`);
    for(const member of enumerate(CAPS[team])) {
      viewer.execute(`MIA_FIXED_LESSON.setAnswer(${CAPS[team]},${member})`);
      for(let depth=0;depth<=5;depth++){viewer.execute(`MIA_FIXED_LESSON.setDepth(${depth})`);assertGraph(viewer,member,depth,null,result);}
      result.team_answer_cases++;
    }
  }
  assert.equal(result.team_answer_cases,198);
  // Each manual result waits indefinitely: only a second click may filter it.
  for(const capacity of [1,2]) for(const member of enumerate(capacity)) {
    viewer.execute(`MIA_FIXED_LESSON.setAnswer(${capacity},${member})`);
    for(let r=0;r<5;r++) {
      await viewer.click('merge-one');assertGraph(viewer,member,r,'result',result);
      assert.equal(viewer.timers.length,0);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').playing,false);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').busy,false);
      assert.equal(viewer.elements.get('merge-one').textContent,'この結果で候補を絞る');assert.equal(viewer.elements.get('merge-one').disabled,false);
      const waiting=viewer.json('MIA_FIXED_LESSON.snapshot()');viewer.click(['show-overview','show-cards','show-both'][r%3]);
      await Promise.resolve();await Promise.resolve();assert.deepEqual(viewer.json('MIA_FIXED_LESSON.snapshot()'),waiting);assert.equal(viewer.timers.length,0);
      assert.equal([...viewer.elements.get('current-calculations').innerHTML.matchAll(/<details>/g)].length,r);
      const filter=viewer.click('merge-one');assertGraph(viewer,member,r,'removing',result);
      assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').playing,false);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').busy,true);
      assert(viewer.elements.get('merge-one').disabled);assert(viewer.elements.get('animate-all').disabled);assert.equal(viewer.timers.length,1);
      viewer.click('merge-one');viewer.click('animate-all');assert.equal(viewer.timers.length,1);
      const fading=viewer.json('MIA_FIXED_LESSON.snapshot()');viewer.click(['show-overview','show-cards','show-both'][(r+1)%3]);assert.deepEqual(viewer.json('MIA_FIXED_LESSON.snapshot()'),fading);
      assert.equal(viewer.timers[0].ms,1700);await viewer.tick();await filter;assertGraph(viewer,member,r+1,null,result);
      assert.equal(viewer.timers.length,0);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').busy,false);result.manual_submission_cases++;
    }
    if(capacity===2) result.two_member_examples++;else result.single_member_examples++;
  }
  assert.equal(result.manual_submission_cases,330);assert.equal(result.single_member_examples,11);assert.equal(result.two_member_examples,55);
  // The explicit automatic-playback button advances all three timed stages.
  for(const capacity of [1,2]) for(const member of enumerate(capacity)) {
    viewer.execute(`MIA_FIXED_LESSON.setAnswer(${capacity},${member})`);const playback=viewer.click('animate-all');
    for(let r=0;r<5;r++) {
      assertGraph(viewer,member,r,'result',result);assert(viewer.json('MIA_FIXED_LESSON.snapshot()').playing);assert.equal(viewer.timers[0].ms,1600);
      await viewer.tick();assertGraph(viewer,member,r,'removing',result);assert.equal(viewer.timers[0].ms,1700);
      await viewer.tick();assertGraph(viewer,member,r+1,null,result);assert.equal(viewer.timers[0].ms,1000);
      await viewer.tick();result.animated_submission_cases++;
    }
    await playback;assert.equal(viewer.timers.length,0);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').playing,false);
  }
  assert.equal(result.animated_submission_cases,330);
  viewer.execute('MIA_FIXED_LESSON.setAnswer(2,5)');await viewer.click('merge-one');assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').results[0],0);
  assert.equal(pairPaths(viewer.elements.get('pair-lines').innerHTML).length,55);assert.equal(viewer.timers.length,0);
  const zero=viewer.click('merge-one');assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').removedMasks.length,40);
  await viewer.tick();await zero;const zeroState=viewer.json('MIA_FIXED_LESSON.snapshot()');assert.equal(zeroState.remainingMasks.length,15);
  zeroState.remainingMasks.forEach(mask=>assert.equal(mask & QUERIES[0],0));assert.equal(zeroState.activeIDs.length,6);
  result.controls_checked.push('all 11 singles and 55 pairs across six prefix lengths and all six Teams','330 manual results create no timer and wait for a second click before filtering','all 330 manual filters stop after exactly one fade, with no next result scheduled','all 330 automatic result / filter / next-result steps independently checked','future results remain unrevealed; calculations include applied submissions only','0-result removes all 40 pairs containing an in-set record; 15 pairs and 6 records remain','excluded record rows stay faint in their original positions, with IDs and all five bits retained');
  viewer.execute('MIA_FIXED_LESSON.setDepth(0)');
  viewer.elements.get('pair-lines').onclick({target:{closest(){return {dataset:{pairMask:'3'}};}}});assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').focusedPair,3);
  assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').remainingMasks.length,55);
  let prevented=false;viewer.elements.get('pair-lines').onkeydown({key:'Enter',preventDefault(){prevented=true;},target:{closest(){return {dataset:{pairMask:'3'}};}}});assert(prevented);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').focusedPair,null);
  viewer.execute('MIA_FIXED_LESSON.setDepth(1)');assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.focusPair(3)'),/現在残る/);
  viewer.select('answer-record-1',675);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[2],1028);
  viewer.select('answer-record-2',675);assert(!viewer.elements.get('lesson-error').hidden);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[2],1028);
  assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setAnswer(1,3)'),/所属人数分/);assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setAnswer(3,7)'),/所属人数分/);assert.throws(()=>viewer.execute('MIA_FIXED_OVERVIEW.selectTeam(6)'),/Team/);
  viewer.click('reset-overview');assert.deepEqual(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships,[2,129,4,24,32,320]);assertGraph(viewer,129,0,null,result);
  viewer.elements.get('overview-table').onclick({target:{closest(){return {dataset:{overviewTeam:'3'}};}}});assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,3);
  viewer.elements.get('overview-table').onclick({target:{closest(){return {dataset:{overviewCellTeam:'5',overviewCellRound:'4'}};}}});assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,5);assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedRound,4);
  const beforeViewSwitch=viewer.json('MIA_FIXED_LESSON.snapshot()'), beforeOverview=viewer.json('MIA_FIXED_OVERVIEW.snapshot()');
  for(const [id,view] of [['show-overview','overview'],['show-cards','cards'],['show-both','both']]) {
    viewer.click(id);assert.equal(viewer.elements.get('lesson-workspace').attributes['data-view'],view);assert.deepEqual(viewer.json('MIA_FIXED_LESSON.snapshot()'),beforeViewSwitch);
    const state=viewer.json('MIA_FIXED_OVERVIEW.snapshot()');assert.deepEqual({...state,view:beforeOverview.view},beforeOverview);assertOverview(viewer,result);
  }
  assert.throws(()=>viewer.execute('MIA_FIXED_OVERVIEW.setView("invalid")'),/表示/);
  viewer.click('reset-overview');await viewer.click('merge-one');viewer.click('spread-all');assert.equal(viewer.timers.length,0);assertGraph(viewer,129,0,null,result);
  await viewer.click('merge-one');const resetDuringFade=viewer.click('merge-one');viewer.execute('MIA_FIXED_OVERVIEW.selectTeam(3)');await viewer.tick();await resetDuringFade;
  assertGraph(viewer,24,0,null,result);viewer.click('reset-overview');
  for(let team=0;team<6;team++) {
    viewer.select('lesson-team',data.teams[team]);const st=viewer.json('MIA_FIXED_LESSON.snapshot()');assert.equal(st.cap,CAPS[team]);
    assert.equal(viewer.elements.get('answer-record-2-wrap').hidden,CAPS[team]===1);assert.equal(viewer.elements.get('tab-single').attributes['aria-selected'],String(CAPS[team]===1));
    assert.equal(viewer.elements.get('tab-pair').attributes['aria-selected'],String(CAPS[team]===2));assertGraph(viewer,st.member,0,null,result);
  }
  viewer.select('lesson-team',11);viewer.select('answer-record-1',675);assertGraph(viewer,1024,0,null,result);
  viewer.click('tab-pair');assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,5);
  viewer.click('tab-single');assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,2);assertGraph(viewer,1024,0,null,result);
  viewer.execute('MIA_FIXED_LESSON.setDepth(5)');assertGraph(viewer,1024,5,null,result);assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.focusPair(3)'),/現在残る線/);viewer.click('reset-overview');
  // Speed changes keep a visible result pending; manual reading has no timer.
  for(const [name,durations] of [['slow',[1600,1700,1000]],['normal',[800,900,500]],['fast',[360,680,200]]]) {
    viewer.select('playback-speed',name);viewer.click('spread-all');await viewer.click('merge-one');assert.equal(viewer.timers.length,0);
    const filter=viewer.click('merge-one');assert.equal(viewer.timers[0].ms,durations[1]);assert.equal(viewer.elements.get('pair-stage').style['--fade-ms'],durations[1]+'ms');await viewer.tick();await filter;
    assert.deepEqual(viewer.json('MIA_FIXED_LESSON.timing()'),{result:durations[0],fade:durations[1],gap:durations[2]});
    viewer.click('spread-all');const auto=viewer.click('animate-all');assert.equal(viewer.timers[0].ms,durations[0]);await viewer.tick();assert.equal(viewer.timers[0].ms,durations[1]);
    await viewer.tick();assert.equal(viewer.timers[0].ms,durations[2]);viewer.click('animate-all');await viewer.tick();await auto;assertGraph(viewer,129,1,null,result);
  }
  viewer.select('playback-speed','slow');viewer.click('spread-all');await viewer.click('merge-one');viewer.select('playback-speed','normal');assertGraph(viewer,129,0,'result',result);assert.equal(viewer.timers.length,0);
  const speedDuringFade=viewer.click('merge-one');viewer.select('playback-speed','slow');assertGraph(viewer,129,0,'result',result);await viewer.tick();await speedDuringFade;assertGraph(viewer,129,0,'result',result);
  viewer.click('tab-single');assertGraph(viewer,2,0,null,result);assert.equal(viewer.timers.length,0);viewer.click('reset-overview');
  // Stop automatic playback in either reveal or fade, then continue manually.
  for(const stage of ['result','removing']) {
    viewer.click('reset-overview');const auto=viewer.click('animate-all');if(stage==='removing')await viewer.tick();
    assertGraph(viewer,129,0,stage,result);viewer.click('animate-all');assertGraph(viewer,129,0,'result',result);assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').playing,false);
    await viewer.tick();await auto;assertGraph(viewer,129,0,'result',result);assert.equal(viewer.timers.length,0);
    const handoff=viewer.click('merge-one');assertGraph(viewer,129,0,'removing',result);await viewer.tick();await handoff;assertGraph(viewer,129,1,null,result);assert.equal(viewer.timers.length,0);
  }
  // Starting automatic playback from a manual result does not skip that submission.
  viewer.click('reset-overview');await viewer.click('merge-one');const resumed=viewer.click('animate-all');assertGraph(viewer,129,0,'result',result);
  for(let r=0;r<5;r++){await viewer.tick();assertGraph(viewer,129,r,'removing',result);await viewer.tick();assertGraph(viewer,129,r+1,null,result);await viewer.tick();}
  await resumed;assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').depth,5);
  const replay=viewer.click('animate-all');assert.equal(viewer.json('MIA_FIXED_LESSON.snapshot()').depth,0);await viewer.tick();assertGraph(viewer,129,0,'result',result);
  viewer.click('animate-all');await viewer.tick();await replay;assertGraph(viewer,129,0,'result',result);
  viewer.click('unmerge-one');assertGraph(viewer,129,0,null,result);assert.equal(viewer.timers.length,0);
  await viewer.click('merge-one');const one=viewer.click('merge-one');await viewer.tick();await one;await viewer.click('merge-one');assertGraph(viewer,129,1,'result',result);
  viewer.click('unmerge-one');assertGraph(viewer,129,1,null,result);viewer.click('unmerge-one');assertGraph(viewer,129,0,null,result);
  await viewer.click('merge-one');const backDuringFade=viewer.click('merge-one');viewer.click('unmerge-one');await viewer.tick();await backDuringFade;assertGraph(viewer,129,0,null,result);
  const reducedViewer=simulatedViewer(true);await reducedViewer.click('merge-one');assert.equal(reducedViewer.timers.length,0);const reducedFilter=reducedViewer.click('merge-one');assert.equal(reducedViewer.timers[0].ms,200);await reducedViewer.tick();await reducedFilter;
  reducedViewer.click('reset-overview');const reduced=reducedViewer.click('animate-all');
  for(let r=0;r<5;r++){for(const duration of [1600,200,1000]){assert.equal(reducedViewer.timers[0].ms,duration);await reducedViewer.tick();}}await reduced;
  assert.equal(reducedViewer.json('MIA_FIXED_LESSON.snapshot()').remainingMasks.length,1);
  viewer.click('open-help');assert.equal(viewer.elements.get('help-overlay').hidden,false);viewer.events.keydown({key:'Escape'});assert.equal(viewer.elements.get('help-overlay').hidden,true);
  result.controls_checked.push('manual back undoes a pending result before undoing an applied submission','stopping automatic playback or changing speed keeps the current result for manual filtering','automatic continuation from a manually revealed result preserves that submission','filtering locks repeated manual/automatic clicks until its one animation completes','old timers cannot update results after reset, Team/speed change, pause or back','overview-only/candidates-only/both views and all six Team/ID selectors stay synchronized','three speeds retained for automatic reading/fading/gaps and manual fading only','reduced-motion manual and automatic playback; help dialog open/Escape/focus');
  result.geometry_sizes_checked=[[320,180],[520,330],[780,500]];
  for(const [width,height] of result.geometry_sizes_checked) {
    const g=viewer.json(`MIA_FIXED_LESSON.geometry(${width},${height})`);assert.equal(g.nodes.length,11);assert.equal(g.edges.length,55);
    const unique=new Set();
    g.edges.forEach(edge=>{
      assert.equal(pc(edge.mask),2);assert(edge.a<edge.b);assert.equal(edge.mask,(1<<edge.a)|(1<<edge.b));unique.add(edge.mask);
      assert.equal(edge.start.id,IDS[edge.a]);assert.equal(edge.end.id,IDS[edge.b]);
      assert(Math.abs(edge.start.y-height*(edge.a+.5)/11)<1e-9);assert(Math.abs(edge.end.y-height*(edge.b+.5)/11)<1e-9);
      for(let sample=0;sample<=20;sample++){const t=sample/20,u=1-t,x=u**3*edge.start.x+3*u*u*t*edge.control+3*u*t*t*edge.control+t**3*edge.end.x,y=u**3*edge.start.y+3*u*u*t*edge.start.y+3*u*t*t*edge.end.y+t**3*edge.end.y;assert(x>=0 && x<=width);assert(y>=0 && y<=height);}
    });assert.equal(unique.size,55);
  }
  const style=html.match(/<style>([\s\S]*?)<\/style>/)[1];assert(style.includes('height:100dvh'));assert(style.includes('grid-template-rows:repeat(11,minmax(0,1fr))'));assert(style.includes('.pair-record.excluded{opacity:.28;'));assert(!style.includes('.pair-record.gone'));assert(style.includes('@keyframes row-disappear{0%{opacity:1}100%{opacity:.28}}'));assert(style.includes('.pair-record.fading{animation:none!important;opacity:.28}'));
  assert(style.includes('.view-toggle{display:flex;flex-wrap:wrap;'));assert(style.includes('.lesson-workspace .overview-panel[hidden],.lesson-workspace .cards-panel[hidden]{display:none!important}'));assert(style.includes('.lesson-workspace[data-view="both"]{grid-template-columns:'));assert(style.includes('grid-template-rows:minmax(0,1fr) minmax(0,1fr)'));assert(style.includes('@keyframes edge-disappear'));assert(style.includes('@media(prefers-reduced-motion:reduce)'));
  assert.equal(11*10/2,55);assert.equal(6*5/2,15);assert.equal(5*6,30);assert.equal(5*4/2,10);
  result.formulas_checked.push('55 unordered pairs, represented by exactly 55 distinct edges','first fixed submission partitions 55 pairs into 15/30/10 for results 0/1/2','all later edge sets use every previously revealed-and-applied result','stable record-row centers match SVG endpoints at all tested sizes; excluded row opacity remains 0.28 in both animation and reduced-motion CSS');
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
  viewer.execute('MIA_FIXED_LESSON.setAnswer(2,1025);MIA_FIXED_LESSON.setDepth(0)');
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
module.exports = {simulatedViewer};
if (require.main === module) main().catch(error => {console.error(error); process.exitCode = 1;});
