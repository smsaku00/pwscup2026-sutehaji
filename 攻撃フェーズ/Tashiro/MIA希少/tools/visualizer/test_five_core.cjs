#!/usr/bin/env node
'use strict';
// Replay every saved single-Team membership and independently check the
// composed score alphabet. This is not a replay of all joint placements.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const five = require('./five_core.js');
const tree = require('./tree_core.js');
const root = path.resolve(__dirname, '../..');
const option = (name, fallback) => {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : path.resolve(process.argv[index + 1]);
};
const sourcePath = option('--source', path.join(root, 'docs/検証資料/追加5回の全候補確認.json'));
const demoPath = option('--demo', null);
const policy = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
const encoding = JSON.parse(fs.readFileSync(path.join(root, 'analysis/full_depth4_20261007/universal_score_encoding.json'), 'utf8'));
const pc = mask => mask.toString(2).replace(/0/g, '').length;
const documentedTruth = [64, 1025, 2, 260, 512, 24];
const documentedMeans = ['233/18000', '169/9000', '7/2000', '553/27000', '11/6000'];
// Derive a reference trace directly from the saved policy and documented
// hypothetical membership, without requiring a duplicated demonstration file.
function documentedDemo() {
  const nodes = policy.caps.map(cap => policy.policies[cap]);
  const states = policy.caps.map(referenceCandidates), trace = [];
  for (let i = 0; i < 5; i++) {
    const responses = [], teams = [];
    for (let t = 0; t < 6; t++) {
      const node = nodes[t], q = Object.hasOwn(node, 'member_mask') ? 0 : node.query_mask;
      const h = pc(documentedTruth[t] & q), before = states[t].length;
      states[t] = states[t].filter(member => pc(member & q) === h);
      if (!Object.hasOwn(node, 'member_mask')) nodes[t] = node.children[h];
      responses.push(h);
      teams.push({ team: policy.team_order[t], query_mask: q, selected_ids: tree.maskIDs(q),
        before, after: states[t].length, padding: policy.total_high[t] - pc(q), remaining_masks: states[t].slice() });
    }
    trace.push({ additional_query: i + 1, submission_round: i + 3, responses, teams, mean_exact: documentedMeans[i] });
  }
  return { truth_masks: documentedTruth, trace };
}
const demo = demoPath ? JSON.parse(fs.readFileSync(demoPath, 'utf8')) : documentedDemo();
assert.deepEqual(demo.truth_masks, documentedTruth);
const result = {
  status: 'passed', scope: 'All 11 one-member and 55 two-member states, plus score-alphabet and disjoint-placement composition checks',
  states_replayed: 0, questions_checked: 0, leaves_checked: 0, maximum_depth_by_capacity: {}, depth_distribution_by_capacity: {},
  padding_checks: 0, score_patterns: 0, mean_collisions: 0,
  demo_team10_remaining: [55], demo_means_exact: [],
  initial_reachable_joint_patterns: 0, disjoint_search_random_cases: 0,
};
assert.deepEqual(policy.caps, five.CAPS);
assert.deepEqual(policy.candidate_ids, tree.IDS);
assert.deepEqual(policy.total_high, encoding.total_high);
assert.equal(policy.compositional_guarantee_max_additional_queries, 5);
assert.equal(policy.composition_covers_disjoint_states, 2494800);

function referenceCandidates(cap) {
  const masks = [];
  for (let m = 1; m <= 2047; m++) if (pc(m) === cap) masks.push(m);
  return masks;
}
function checkPolicy(node, states, cap, depth, distribution) {
  if (Object.hasOwn(node, 'member_mask')) {
    assert.equal(states.length, 1);
    assert.equal(node.member_mask, states[0]);
    assert.equal(pc(node.member_mask), cap);
    assert(five.complete(node));
    assert.equal(five.query(node), 0);
    distribution[depth] = (distribution[depth] || 0) + 1;
    result.leaves_checked++;
    return;
  }
  assert(!five.complete(node));
  assert.equal(node.before, states.length);
  assert.equal(five.query(node), node.query_mask);
  assert(Number.isInteger(node.query_mask) && node.query_mask >= 0 && node.query_mask <= 2047);
  assert(pc(node.query_mask) <= 5);
  assert.equal(node.query_letters, 'ABCDEFGHIJK'.split('').filter((_, i) => node.query_mask & (1 << i)).join(''));
  for (let t = 0; t < 6; t++) {
    if (policy.caps[t] !== cap) continue;
    const padding = policy.total_high[t] - pc(node.query_mask);
    assert(padding >= 1 && padding <= 13);
    result.padding_checks++;
  }
  const referenceGroups = new Map();
  for (const member of states) {
    const hits = pc(member & node.query_mask);
    if (!referenceGroups.has(hits)) referenceGroups.set(hits, []);
    referenceGroups.get(hits).push(member);
  }
  assert.deepEqual([...five.groups(states, node.query_mask)], [...referenceGroups]);
  assert.deepEqual(Object.keys(node.children).map(Number).sort(), [...referenceGroups.keys()].sort());
  assert(Math.max(...[...referenceGroups.values()].map(v => v.length)) < states.length);
  result.questions_checked++;
  for (const [hits, subset] of referenceGroups) checkPolicy(node.children[hits], subset, cap, depth + 1, distribution);
}
for (const cap of [1, 2]) {
  const states = five.candidates(cap);
  assert.deepEqual([...states].sort((a, b) => a - b), referenceCandidates(cap));
  assert.equal(new Set(states).size, cap === 1 ? 11 : 55);
  const distribution = {};
  checkPolicy(policy.policies[cap], states, cap, 0, distribution);
  assert.deepEqual(distribution, policy.single_team[cap].depth_distribution);
  const maxDepth = Math.max(...Object.keys(distribution).map(Number));
  assert.equal(maxDepth, cap === 1 ? 4 : 5);
  result.maximum_depth_by_capacity[cap] = maxDepth;
  result.depth_distribution_by_capacity[cap] = distribution;
  for (const truth of states) {
    const hits = [];
    let current = five.navigateTeam(policy, cap, hits);
    while (!current.complete) {
      hits.push(pc(truth & current.query));
      current = five.navigateTeam(policy, cap, hits);
      assert(current.candidates.includes(truth));
      assert(hits.length <= 5);
    }
    assert.deepEqual(current.candidates, [truth]);
    // Finished Teams submit an empty candidate mask until other Teams finish.
    while (hits.length < 5) { hits.push(0); current = five.navigateTeam(policy, cap, hits); }
    assert(current.complete);
    assert.equal(current.query, 0);
    assert.deepEqual(current.candidates, [truth]);
    result.states_replayed++;
  }
}
assert.equal(result.states_replayed, 66);
assert.equal(result.questions_checked, 49);
assert.equal(result.leaves_checked, 66);
assert.deepEqual(five.navigateTeam(policy.policies, 2, [1, 1, 0, 1, 0]).candidates, [1025]);
assert.deepEqual(five.navigateTeam(policy.policies[2], 2, [1, 1, 0, 1, 0]).candidates, [1025]);
assert.throws(() => five.candidates(0));
assert.throws(() => five.navigateTeam(policy, 1, [2]));
assert.throws(() => five.navigateTeam(policy, 2, [0, 0, 0, 0, 0, 0]));
assert.throws(() => five.navigateTeam(policy, 2, [1.5]));
const shortLeaf = five.navigateTeam(policy, 2, [2, 1, 1]);
assert(shortLeaf.complete);
assert(five.navigateTeam(policy, 2, [2, 1, 1, 0, 0]).complete);
assert.throws(() => five.navigateTeam(policy, 2, [2, 1, 1, 1]));

// Exact arithmetic from independent ROC interpolation at FPR=1/100.
const gcd = (a, b) => { while (b) [a, b] = [b, a % b]; return a < 0n ? -a : a; };
const fraction = (n, d = 1n) => { n = BigInt(n); d = BigInt(d); assert(d !== 0n); const g = gcd(n, d); return { n: n / g, d: d / g }; };
const add = (a, b) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a, b) => fraction(a.n * b.n, a.d * b.d);
const divide = (a, b) => fraction(a.n * b.d, a.d * b.n);
const exact = a => a.d === 1n ? String(a.n) : `${a.n}/${a.d}`;
function meanFromROC(hits) {
  let sum = fraction(0);
  for (let t = 0; t < 6; t++) {
    const x = fraction(policy.total_high[t] - hits[t], 24 - policy.caps[t]);
    const y = fraction(hits[t], policy.caps[t]);
    const target = fraction(1, 100);
    assert(x.n * target.d > target.n * x.d);
    sum = add(sum, divide(multiply(y, target), x));
  }
  return divide(sum, fraction(6));
}
const values = [];
for (const [codeString, integer] of Object.entries(encoding.integer_values)) {
  const code = Number(codeString), hits = tree.counts(code), mean = meanFromROC(hits);
  assert.deepEqual(mean, fraction(integer, encoding.mean_denominator));
  assert.equal(tree.decodeMean(exact(mean), encoding), code);
  assert.equal(tree.decodeMean(Number(mean.n) / Number(mean.d), encoding), code);
  values.push(mean);
  result.score_patterns++;
}
assert.equal(new Set(values.map(exact)).size, 216);
values.sort((a, b) => a.n * b.d < b.n * a.d ? -1 : 1);
let minGap = null;
for (let i = 1; i < values.length; i++) {
  const gap = add(values[i], fraction(-values[i - 1].n, values[i - 1].d));
  if (!minGap || gap.n * minGap.d < minGap.n * gap.d) minGap = gap;
}
assert.equal(exact(minGap), '1/54000');
result.minimum_mean_gap_exact = exact(minGap);

function assertDisjoint(row, sets) {
  assert(row && row.length === 6);
  let used = 0;
  row.forEach((member, t) => { assert(sets[t].includes(member)); assert(!(member & used)); used |= member; });
  assert.equal(pc(used), 9);
}
const initial = policy.caps.map(cap => five.candidates(cap));
assertDisjoint(five.findDisjoint(initial), initial);
const truthSets = demo.truth_masks.map(member => [member]);
assert.deepEqual(five.findDisjoint(truthSets), demo.truth_masks);
const conflict = truthSets.map(s => s.slice()); conflict[2] = conflict[0].slice();
assert.equal(five.findDisjoint(conflict), null);
assert.equal(five.findDisjoint([[], ...initial.slice(1)]), null);
assert.throws(() => five.findDisjoint(initial.slice(1)));
assert.throws(() => five.findDisjoint([[0], ...initial.slice(1)]));
const allZero = policy.caps.map(cap => five.navigateTeam(policy, cap, [0]).candidates);
assert.equal(five.findDisjoint(allZero), null);
for (const code of Object.keys(encoding.integer_values).map(Number)) {
  const sets = tree.counts(code).map((h, t) => five.navigateTeam(policy, policy.caps[t], [h]).candidates);
  const found = five.findDisjoint(sets);
  if (found) { assertDisjoint(found, sets); result.initial_reachable_joint_patterns++; }
}
// Independent small exhaustive search tests memoization and pruning outcomes.
function brute(sets, t = 0, used = 0) {
  if (t === 6) return true;
  return sets[t].some(m => !(m & used) && brute(sets, t + 1, used | m));
}
let seed = 20261008;
const random = n => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
for (let i = 0; i < 200; i++) {
  const sets = initial.map(options => Array.from({ length: 1 + random(4) }, () => options[random(options.length)]));
  const found = five.findDisjoint(sets);
  assert.equal(!!found, brute(sets));
  if (found) assertDisjoint(found, sets);
  result.disjoint_search_random_cases++;
}

const histories = policy.caps.map(() => []);
for (const [index, step] of demo.trace.entries()) {
  assert.equal(step.additional_query, index + 1);
  assert.equal(step.submission_round, index + 3);
  const mean = meanFromROC(step.responses);
  assert.equal(exact(mean), step.mean_exact);
  const code = step.responses.reduce((sum, h, t) => sum + h * 3 ** t, 0);
  assert.equal(tree.decodeMean(step.mean_exact, encoding), code);
  result.demo_means_exact.push(exact(mean));
  for (let t = 0; t < 6; t++) {
    const before = five.navigateTeam(policy, policy.caps[t], histories[t]);
    assert.equal(before.query, step.teams[t].query_mask);
    assert.equal(before.candidates.length, step.teams[t].before);
    assert.equal(policy.total_high[t] - pc(before.query), step.teams[t].padding);
    assert.deepEqual(tree.maskIDs(before.query), step.teams[t].selected_ids);
    assert.equal(pc(demo.truth_masks[t] & before.query), step.responses[t]);
    histories[t].push(step.responses[t]);
    const after = five.navigateTeam(policy, policy.caps[t], histories[t]);
    assert.deepEqual([...after.candidates].sort((a, b) => a - b), [...step.teams[t].remaining_masks].sort((a, b) => a - b));
    assert.equal(after.candidates.length, step.teams[t].after);
    assert(after.candidates.includes(demo.truth_masks[t]));
  }
  result.demo_team10_remaining.push(five.navigateTeam(policy, 2, histories[1]).candidates.length);
  const sets = policy.caps.map((cap, t) => five.navigateTeam(policy, cap, histories[t]).candidates);
  assertDisjoint(five.findDisjoint(sets), sets);
}
assert.deepEqual(result.demo_team10_remaining, [55, 28, 13, 5, 2, 1]);
assert.deepEqual(policy.caps.map((cap, t) => five.navigateTeam(policy, cap, histories[t]).candidates[0]), demo.truth_masks);
assert.equal(result.demo_means_exact[0], '233/18000');
assert.deepEqual(result.demo_means_exact, documentedMeans);
result.source_sha256 = crypto.createHash('sha256').update(fs.readFileSync(sourcePath)).digest('hex');
if (demoPath) result.demo_sha256 = crypto.createHash('sha256').update(fs.readFileSync(demoPath)).digest('hex');
result.documented_demo_membership_masks = documentedTruth;
result.core_sha256 = crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname, 'five_core.js'))).digest('hex');
if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_five_core.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
