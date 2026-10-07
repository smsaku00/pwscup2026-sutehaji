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
  const tick=async()=>{assert(timers.length);timers.shift().callback();await Promise.resolve();await Promise.resolve();};
  return {elements,execute,json,select,click,downloads,timers,tick,events,document};
}
function assertOverview(viewer,result) {
  const state=viewer.json('MIA_FIXED_OVERVIEW.snapshot()'), lesson=viewer.json('MIA_FIXED_LESSON.values()');
  let used=0;
  state.memberships.forEach((member,t)=>{assert.equal(pc(member),CAPS[t]);assert.equal(used & member,0);used|=member;});
  assert.equal(pc(used),9);
  const responses=QUERIES.map(query=>state.memberships.map(member=>pc(member & query)));
  assert.deepEqual(state.responses,responses);
  const columns=CAPS.map((_,t)=>responses.map(row=>row[t]));
  assert.deepEqual(state.columns,columns);assert.deepEqual(state.rowSequences,responses.map(row=>row.join('')));
  assert.deepEqual(state.meanFractions,responses.map(exactMean));
  const members=columns.map((column,t)=>enumerate(CAPS[t]).filter(candidate=>QUERIES.every((query,r)=>pc(candidate & query)===column[r])));
  members.forEach((set,t)=>assert.deepEqual(set,[state.memberships[t]]));
  assert.deepEqual(state.decodedMembers,state.memberships);
  const t=state.selectedTeam,r=state.selectedRound;
  assert.equal(CAPS[t],lesson.cap);assert.equal(state.memberships[t],lesson.masks[lesson.cap]);
  assert.equal(state.depth,lesson.depth);
  const markup=viewer.elements.get('overview-table').innerHTML;
  const values=[...markup.matchAll(/<td data-overview-cell-team="(\d+)" data-overview-cell-round="(\d+)" data-hit="(\d+)"[^>]*>(\d+)<\/td>/g)];
  assert.equal(values.length,30);
  values.forEach(value=>{const team=Number(value[1]),round=Number(value[2]);assert.equal(Number(value[3]),responses[round][team]);assert.equal(Number(value[4]),responses[round][team]);result.overview_cell_checks++;});
  const columnMarkup=[...markup.matchAll(/<td data-overview-column="(\d+)"[^>]*>([012]{5})<\/td>/g)];
  assert.equal(columnMarkup.length,6);columnMarkup.forEach(value=>assert.equal(value[2],columns[Number(value[1])].join('')));
  const decoded=[...markup.matchAll(/<td data-overview-member="(\d+)"[^>]*>([^<]+)<\/td>/g)];
  assert.equal(decoded.length,6);decoded.forEach(value=>assert.equal(value[2],label(state.memberships[Number(value[1])]).replaceAll('・','＋')));
  assert(viewer.elements.get('row-reader').innerHTML.includes(state.rowSequences[r]));
  assert(viewer.elements.get('row-reader').innerHTML.includes(state.meanFractions[r]));
  assert(viewer.elements.get('column-reader').innerHTML.includes(columns[t].join(' ↓ ')));
  assert(viewer.elements.get('selected-team-caption').textContent.includes(`Team ${data.teams[t]}`));
  const trial=viewer.json('MIA_FIXED_LESSON.trialSnapshot()');assert.deepEqual(trial.target,columns[t]);
  const headerAnswer=viewer.elements.get('target-row').innerHTML;assert(headerAnswer.includes('答え'));
  const headerDigits=[...headerAnswer.matchAll(/data-round="(\d+)" data-value="([012])">([012])<\/span>/g)];
  assert.equal(headerDigits.length,5);headerDigits.forEach((m,i)=>{assert.equal(Number(m[1]),i);assert.equal(Number(m[2]),columns[t][i]);assert.equal(Number(m[3]),columns[t][i]);});
  for(let i=0;i<5;i++) assert.equal(viewer.elements.get(`record-heading-${i+1}`).classList.contains('row-focus'),i===r);
  result.overview_states_checked++;
}

const recordCards=markup=>[...markup.matchAll(/<button type="button" class="record-card([^\"]*)"([^>]*)>([\s\S]*?)<\/button>/g)].map(m=>({classes:m[1],attrs:attrs(m[2]),body:m[3]}));
function assertRecords(viewer,cap,member,depth,result) {
  const rows=independentTrace(cap,member), api=viewer.json(`MIA_FIXED_LESSON.trace(${cap},${member})`);
  rows.forEach((row,i)=>{
    assert.equal(api[i].query,row.query);assert.equal(api[i].hit,row.hit);assert.equal(api[i].complete,false);
    assert.deepEqual(sorted(api[i].before),row.before);assert.deepEqual(sorted(api[i].after),row.after);
    row.groups.forEach((group,h)=>assert.deepEqual(sorted(api[i].groups[h]),group));
    const calculation=viewer.elements.get('current-calculations').innerHTML;
    assert(calculation.includes(`対象ID：${label(QUERIES[i])}`));
    row.groups.forEach((group,h)=>assert(calculation.includes(`<b>${h}人：${group.length}${cap===1?'候補':'組'}</b>`)));
    if(row.before.length===1) result.fixed_submissions_after_early_certainty++;
    result.submission_rows_checked++;
  });
  const kept=enumerate(cap).filter(candidate=>rows.slice(0,depth).every(row=>pc(candidate & row.query)===row.hit));
  assert.deepEqual(sorted(viewer.json(`MIA_FIXED_LESSON.remaining(${cap},${member},${depth})`)),kept);
  assert.equal(viewer.elements.get('count-path').textContent,[enumerate(cap).length,...rows.map(row=>row.after.length)].join(' → '));
  const state=viewer.json('MIA_FIXED_LESSON.trialSnapshot()');
  assert.deepEqual(state.selectedIDs,IDS.filter((_,i)=>member>>i&1));
  assert.deepEqual(state.target,signature(member));assert.deepEqual(state.sum,signature(member));
  assert(state.complete);assert(state.allMatch);assert.equal(state.comparedCount,depth);
  const cards=recordCards(viewer.elements.get('record-deck').innerHTML);
  assert.equal(cards.length,11);assert.deepEqual(cards.map(c=>Number(c.attrs['data-record-id'])),IDS);
  cards.forEach(card=>{
    const id=Number(card.attrs['data-record-id']), bit=1<<IDS.indexOf(id), expected=signature(bit);
    assert.equal(card.attrs['data-signature'],expected.join(''));
    const bits=[...card.body.matchAll(/data-round="(\d+)" data-value="([01])"/g)];
    assert.equal(bits.length,5);
    bits.forEach((m,r)=>{assert.equal(Number(m[1]),r);assert.equal(Number(m[2]),expected[r]);result.record_bit_checks++;});
    assert.equal(card.attrs['aria-pressed'],String(state.selectedIDs.includes(id)));
    assert.equal(card.classes.includes('solved'),depth===5 && state.selectedIDs.includes(id));
  });
  const summed=[...viewer.elements.get('sum-result').innerHTML.matchAll(/data-round="(\d+)" data-value="(\d+)" data-checked="([01])" data-match="([01]?)"/g)];
  assert.equal(summed.length,5);summed.forEach((m,r)=>{assert.equal(Number(m[2]),signature(member)[r]);assert.equal(m[3],r<depth?'1':'0');assert.equal(m[4],r<depth?'1':'');result.sum_cell_checks++;});
  if(depth===5) {assert.deepEqual(kept,[member]);assert(viewer.elements.get('trial-status').textContent.includes('5回とも一致'));}
  assertOverview(viewer,result);
}
async function main() {
  const result={status:'passed',environment:'Node VM simulated DOM, independent record/pair enumeration, controlled animation timers and numeric layout geometry',
    html_bytes:Buffer.byteLength(html),html_sha256:crypto.createHash('sha256').update(html).digest('hex'),
    one_member_examples:0,two_member_examples:0,example_prefixes_checked:0,submission_rows_checked:0,record_bit_checks:0,sum_cell_checks:0,
    fixed_submissions_after_early_certainty:0,overview_states_checked:0,overview_cell_checks:0,formulas_checked:[],controls_checked:[],mean_decodes_checked:[],negative_cases:[],visual_browser_validation:false};
  const viewer=simulatedViewer();
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.values()'),{cap:1,depth:0,masks:{'1':128,'2':3},playing:false});
  assert.equal(viewer.elements.get('answer-record-2-wrap').hidden,true);assert.equal(viewer.elements.get('help-overlay').hidden,true);
  assert.deepEqual(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships,DEMO);assertOverview(viewer,result);
  for(const cap of [1,2]) for(const member of enumerate(cap)) {
    viewer.execute(`MIA_FIXED_LESSON.setAnswer(${cap},${member})`);
    const selected=IDS.filter((_,i)=>member>>i&1);
    assert.equal(viewer.elements.get('answer-record-1').value,String(selected[0]));
    if(cap===2) assert.equal(viewer.elements.get('answer-record-2').value,String(selected[1]));
    for(let depth=0;depth<=5;depth++) {viewer.execute(`MIA_FIXED_LESSON.setDepth(${depth})`);assertRecords(viewer,cap,member,depth,result);result.example_prefixes_checked++;}
    viewer.click('find-matching');assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').selectedIDs,selected);
    result[cap===1?'one_member_examples':'two_member_examples']++;
  }
  assert.equal(result.example_prefixes_checked,396);
  result.controls_checked.push('all 11 singles and 55 pairs at all six comparison prefixes','eleven visible record cards with five independently checked fixed bits','all selected record sums match the true Team column; complete matches turn selected records green','matching-ID selection searches all full five-entry patterns');
  result.overview_answer_cases=0;
  for(let team=0;team<6;team++) {
    viewer.execute(`MIA_FIXED_OVERVIEW.selectTeam(${team})`);
    for(const member of enumerate(CAPS[team])) {
      viewer.execute(`MIA_FIXED_LESSON.setAnswer(${CAPS[team]},${member});MIA_FIXED_LESSON.setDepth(5)`);
      assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,team);assertOverview(viewer,result);result.overview_answer_cases++;
    }
  }
  assert.equal(result.overview_answer_cases,198);
  viewer.click('reset-overview');
  viewer.elements.get('overview-table').onclick({target:{closest(){return {dataset:{overviewTeam:'4'}};}}});
  assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,4);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').cap,1);
  viewer.elements.get('overview-table').onclick({target:{closest(){return {dataset:{overviewCellTeam:'3',overviewCellRound:'4'}};}}});
  assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedTeam,3);assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedRound,4);assertOverview(viewer,result);
  viewer.elements.get('overview-table').onclick({target:{closest(){return {dataset:{overviewRound:'2'}};}}});
  assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedRound,2);
  viewer.click('show-cards');assert.equal(viewer.elements.get('lesson-workspace').attributes['data-view'],'cards');
  viewer.click('show-overview');assert.equal(viewer.elements.get('lesson-workspace').attributes['data-view'],'overview');
  viewer.click('reset-overview');assertOverview(viewer,result);
  result.controls_checked.push('five six-Team response rows and six five-entry Team columns','all 198 possible answers across all six Teams retain a legal disjoint joint placement','table row/column/cell selections connect to the selected Team and submission heading','narrow-screen view tabs and full reset');
  viewer.execute('MIA_FIXED_LESSON.setAnswer(2,3)');
  const unchanged=viewer.json('MIA_FIXED_OVERVIEW.snapshot()');
  let matched=0;result.trial_pair_cases=0;
  for(const pair of enumerate(2)) {
    const chosen=IDS.filter((_,i)=>pair>>i&1);
    viewer.execute(`MIA_FIXED_LESSON.setTrial(${JSON.stringify(chosen)});MIA_FIXED_LESSON.setDepth(5)`);
    const trial=viewer.json('MIA_FIXED_LESSON.trialSnapshot()');
    assert.deepEqual(trial.sum,signature(pair));assert.deepEqual(trial.target,signature(3));assert.equal(trial.allMatch,pair===3);
    assert.deepEqual(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships,unchanged.memberships);
    const colored=recordCards(viewer.elements.get('record-deck').innerHTML).filter(c=>c.classes.includes('solved'));
    assert.equal(colored.length,pair===3?2:0);assert.equal(viewer.elements.get('trial-status').textContent.includes('不一致'),pair!==3);
    if(trial.allMatch) matched++;result.trial_pair_cases++;
  }
  assert.equal(matched,1);
  viewer.execute('MIA_FIXED_LESSON.setTrial([47,76])');
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').signatures,[[0,1,0,1,1],[1,0,1,1,1]]);
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').sum,[1,1,1,2,2]);
  viewer.elements.get('record-deck').onclick({target:{closest(){return {dataset:{recordId:'47'}};}}});
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').selectedIDs,[76]);assert(viewer.elements.get('animate-all').disabled);assert(viewer.elements.get('merge-one').disabled);
  assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setDepth(1)'),/人数分/);
  viewer.elements.get('record-deck').onclick({target:{closest(){return {dataset:{recordId:'90'}};}}});
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').selectedIDs,[76,90]);
  viewer.elements.get('record-deck').onclick({target:{closest(){return {dataset:{recordId:'115'}};}}});
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').selectedIDs,[90,115]);
  assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setTrial([47,47])'),/異なる/);assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setTrial([47,76,90])'),/人数分/);
  assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setTrial([999])'),/候補/);
  viewer.click('find-matching');assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').selectedIDs,[47,76]);
  assert.deepEqual(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships,unchanged.memberships);
  result.controls_checked.push('all 55 trial pairs against a fixed Team truth; exactly one match and 54 explicit mismatches','trial record clicks do not change the observed Team response or joint truth','incomplete selection disables comparison; duplicate and out-of-range records rejected','01011 + 10111 = 11122 and two overlapping ones count as 2');
  viewer.click('tab-single');viewer.select('answer-record-1',383);
  viewer.click('tab-pair');viewer.execute('MIA_FIXED_LESSON.setAnswer(2,3)');
  viewer.select('answer-record-2',90);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[2],5);
  viewer.select('answer-record-1',90);assert(!viewer.elements.get('lesson-error').hidden);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[2],5);
  viewer.select('answer-record-1',675);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[2],1028);
  assert.equal(viewer.elements.get('answer-record-1').value,'675');assert.equal(viewer.elements.get('answer-record-2').value,'90');
  viewer.click('tab-single');assert.equal(viewer.json('MIA_FIXED_LESSON.values()').masks[1],128);
  assert.throws(()=>viewer.execute('MIA_FIXED_LESSON.setAnswer(2,1)'),/異なる正解/);
  const sumCells=v=>[...v.elements.get('sum-result').innerHTML.matchAll(/<span class="sum-bit ([^"]+)"([^>]*)>([^<]+)<small>/g)].map(m=>({classes:m[1],attrs:attrs(m[2]),display:m[3]}));
  const assertDisplayedSums=(v,depth,member)=>{
    const expected=signature(member), cells=sumCells(v);assert.equal(cells.length,5);
    cells.forEach((cell,r)=>assert.equal(cell.display,r<depth?String(expected[r]):'·'));
  };
  // Complete all five manual animations for each possible one/two-person answer.
  // During movement the newly calculated sum must stay hidden; after movement
  // exactly one more submission is revealed, including 1+1=2 cases.
  result.animated_step_cases=0;
  for(const capacity of [1,2]) for(const member of enumerate(capacity)) {
    viewer.execute(`MIA_FIXED_LESSON.setAnswer(${capacity},${member})`);
    const truth=viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships;
    for(let r=0;r<5;r++) {
      const step=viewer.click('merge-one');
      assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,r);
      assert.equal(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').addingRound,r);
      assert.equal(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').selectedRound,r);
      assertDisplayedSums(viewer,r,member);
      assert(viewer.elements.get('merge-one').disabled);assert.equal(viewer.timers.length,1);
      viewer.click('merge-one');assert.equal(viewer.timers.length,1);
      for(let i=1;i<=capacity;i++) {
        const moving=[...viewer.elements.get(`floating-record-${i}`).innerHTML.matchAll(/class="floating-bit[^\"]* adding-digit" data-round="(\d+)"/g)];
        assert.equal(moving.length,1);assert.equal(Number(moving[0][1]),r);
      }
      assert.equal(viewer.timers[0].ms,680);await viewer.tick();await step;
      assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,r+1);
      assert.equal(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').addingRound,-1);
      assertDisplayedSums(viewer,r+1,member);
      assert.deepEqual(viewer.json('MIA_FIXED_OVERVIEW.snapshot()').memberships,truth);
      result.animated_step_cases++;
    }
  }
  assert.equal(result.animated_step_cases,330);
  assert(viewer.elements.get('merge-one').disabled);viewer.click('unmerge-one');assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,4);
  assertDisplayedSums(viewer,4,1536);
  viewer.click('spread-all');assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,0);
  const play=viewer.click('animate-all');
  for(let r=0;r<5;r++) {
    assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,r);
    assert.equal(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').addingRound,r);
    assert.equal(viewer.timers[0].ms,680);await viewer.tick();
    assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,r+1);
    assert.equal(viewer.timers[0].ms,180);await viewer.tick();
  }
  await play;assert.equal(viewer.json('MIA_FIXED_LESSON.values()').playing,false);
  const replay=viewer.click('animate-all');assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,0);
  assert.equal(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').addingRound,-1);await viewer.tick();
  assert.equal(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').addingRound,0);
  viewer.click('tab-pair');await viewer.tick();await replay;assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,0);
  const stopped=viewer.click('animate-all');viewer.click('animate-all');await viewer.tick();await stopped;
  assert.equal(viewer.json('MIA_FIXED_LESSON.values()').playing,false);assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,0);
  const cancelled=viewer.click('merge-one');viewer.execute('MIA_FIXED_LESSON.setTrial([47,90])');await viewer.tick();await cancelled;
  assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,0);assert.equal(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').addingRound,-1);
  assertDisplayedSums(viewer,0,5);
  const interrupted=viewer.click('merge-one');viewer.click('spread-all');await viewer.tick();await interrupted;
  assert.equal(viewer.json('MIA_FIXED_LESSON.values()').depth,0);assertDisplayedSums(viewer,0,5);
  // A wrong trial still calculates the same sum; it does not replace the answer.
  viewer.execute('MIA_FIXED_LESSON.setAnswer(2,3);MIA_FIXED_LESSON.setTrial([90,115])');
  const mismatch=viewer.click('merge-one');await viewer.tick();await mismatch;
  assert.equal(sumCells(viewer)[0].display,'0');assert(sumCells(viewer)[0].classes.includes('mismatch'));
  assert.deepEqual(viewer.json('MIA_FIXED_LESSON.trialSnapshot()').target,[1,1,1,2,2]);
  viewer.click('open-help');assert.equal(viewer.elements.get('help-overlay').hidden,false);assert.equal(viewer.document.activeElement.id,'close-help');
  viewer.events.keydown({key:'Escape'});assert.equal(viewer.elements.get('help-overlay').hidden,true);assert.equal(viewer.document.activeElement.id,'open-help');
  result.controls_checked.push('one/two person tabs and selected answer order','manual per-submission addition for all 330 steps, revealing each sum only after motion','future sums stay blank; answers remain fixed in the header','automatic mode uses the same per-submission addition as the manual button','busy manual button prevents duplicate runs; replay, pause and cancellation on tab/trial/reset','wrong selected pair animates its own sum and reports mismatch against unchanged answer','help dialog open/Escape/focus restoration');
  const reducedViewer=simulatedViewer(true), reducedPlay=reducedViewer.click('animate-all');
  for(let r=0;r<5;r++) {
    assert.equal(reducedViewer.json('MIA_FIXED_LESSON.values()').depth,r);assert.equal(reducedViewer.timers[0].ms,20);await reducedViewer.tick();
    assert.equal(reducedViewer.json('MIA_FIXED_LESSON.values()').depth,r+1);assert.equal(reducedViewer.timers[0].ms,20);await reducedViewer.tick();
  }
  await reducedPlay;assert.equal(reducedViewer.json('MIA_FIXED_LESSON.values()').playing,false);
  result.controls_checked.push('reduced-motion animation reaches all five comparisons');
  result.geometry_heights_checked=[60,72,78,96,108,160,260,420];
  for(const height of result.geometry_heights_checked) {
    const g=viewer.json(`MIA_FIXED_LESSON.geometry(${height})`);
    assert(g.cardHeight>0);assert.equal(g.positions.length,3);assert(Math.abs(g.target+g.cardHeight-height)<1e-9);
    for(let i=0;i<2;i++) assert(g.positions[i]+g.cardHeight<=g.positions[i+1]);
    viewer.elements.get('overlay-stage').clientHeight=height;viewer.execute('MIA_FIXED_LESSON.setDepth(0);MIA_FIXED_LESSON.place(true)');
    for(let i=1;i<=2;i++) assert.equal(viewer.elements.get(`floating-record-${i}`).style.transform,`translateY(${g.positions[i-1]}px)`);
    viewer.execute('MIA_FIXED_LESSON.setDepth(5)');
    for(let i=1;i<=2;i++) {assert.equal(viewer.elements.get(`floating-record-${i}`).style.transform,`translateY(${g.positions[i-1]}px)`);assert.equal(viewer.elements.get(`floating-record-${i}`).style['--addition-distance'],`${g.target-g.positions[i-1]}px`);}
  }
  const style=html.match(/<style>([\s\S]*?)<\/style>/)[1];
  assert(style.includes('html,body{height:100%;margin:0;overflow:hidden}'));assert(style.includes('height:100dvh'));
  assert(style.includes('grid-template-rows:repeat(11,minmax(0,1fr))'));assert(style.includes('@keyframes add-digit'));assert(style.includes('animation:add-digit 620ms'));assert(style.includes('@media(prefers-reduced-motion:reduce)'));
  const staticMarkup=html.split('<script>')[0];assert(!staticMarkup.includes('質問'));assert(!staticMarkup.includes('調査'));assert(!staticMarkup.includes('glass-grid-'));assert(staticMarkup.includes('55は2人の選び方の数'));
  assert(staticMarkup.indexOf('id="target-row"')<staticMarkup.indexOf('id="record-deck"'));
  assert.equal((staticMarkup.match(/id="target-row"/g)||[]).length,1);
  assert.equal(11*10/2,55);assert.equal(6*5/2+5*6+5*4/2,55);
  result.formulas_checked.push('11 choose 2 = 55, with eleven visible per-record cards','first pair partition 15+30+10=55','all later counts use previously surviving candidates');
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
  viewer.execute('MIA_FIXED_LESSON.setAnswer(1,1);MIA_FIXED_LESSON.setAnswer(2,1025);MIA_FIXED_LESSON.setDepth(0)');
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
