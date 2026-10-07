#!/usr/bin/env node
'use strict';
// Independent integer-sum codebook enumeration, not an adaptive-tree replay.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const fixed = require('./fixed_core.js');
const tree = require('./tree_core.js');
const root = path.resolve(__dirname, '../..');
const sourcePath = path.join(root, 'docs/visualizer/resources/nonadaptive_code.json');
const encodingPath = path.join(root, 'analysis/full_depth4_20261007/universal_score_encoding.json');
const paddingPath = path.join(root, 'analysis/full_depth4_20261007/padding_rule.json');
const source = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
const encoding = JSON.parse(fs.readFileSync(encodingPath, 'utf8'));
const padding = JSON.parse(fs.readFileSync(paddingPath, 'utf8'));
const pc = mask => mask.toString(2).replace(/0/g, '').length;
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const result = {
  status: 'passed', strategy: 'nonadaptive',
  scope: 'All 11 singleton and 55 pair integer-sum signatures, every fixed prefix, padding, exact 216-pattern score mapping, and disjoint-Team composition checks. Joint placements are covered compositionally, not replayed exhaustively.',
  states_checked: 0, signature_counts: {}, maximum_prefix_depth: {}, first_unique_depth_distribution: {},
  maximum_candidate_counts_by_prefix: {}, prefix_groups_checked: 0, padding_checks: 0,
  score_patterns: 0, mean_collisions: 0, disjoint_search_random_cases: 0,
  initial_reachable_joint_patterns: 0,
};
assert.equal(source.strategy, 'nonadaptive');
assert.equal(source.provenance.kind, 'new_construction');
assert.equal(source.question_count, 5);
assert.deepEqual(source.candidate_ids, fixed.IDS);
assert.deepEqual(source.candidate_ids, tree.IDS);
assert.deepEqual(source.codewords, fixed.CODEWORDS);
assert.deepEqual(source.query_masks, fixed.QUERY_MASKS);
assert.deepEqual(source.caps, fixed.CAPS);
assert.deepEqual(source.total_high, fixed.HIGH);
assert.deepEqual(source.total_high, encoding.total_high);
assert.deepEqual(source.padding_counts, fixed.PADDING);
assert.deepEqual(source.candidate_ids, padding.candidate_ids);
assert.equal(source.confidence.high, 0.4);
assert.equal(source.confidence.other_rare, 0.1);
assert(padding.rule.includes('Assign 0.4'));
assert(padding.rule.includes('0.1 to the other rare rows'));

// Enumerate members independently by the integer's set-bit count. Derive each
// answer by summing codeword bits, rather than using the core's mask overlap.
function referenceCandidates(cap) {
  return Array.from({ length: 2047 }, (_, index) => index + 1).filter(member => pc(member) === cap);
}
function referenceSignature(member) {
  const members = source.codewords.filter((_, index) => member & 2 ** index);
  return Array.from({ length: 5 }, (_, question) => members.reduce((sum, word) => sum + Math.floor(word / 2 ** question) % 2, 0));
}
function referenceRemaining(cap, hits) {
  return referenceCandidates(cap).filter(member => {
    const sig = referenceSignature(member);
    return hits.every((hit, index) => sig[index] === hit);
  });
}
for (let question = 0; question < 5; question++) {
  let referenceMask = 0;
  const expectedIDs = [];
  source.codewords.forEach((word, index) => {
    if (Math.floor(word / 2 ** question) % 2) { referenceMask += 2 ** index; expectedIDs.push(source.candidate_ids[index]); }
  });
  assert.equal(referenceMask, fixed.QUERY_MASKS[question]);
  assert.equal(pc(referenceMask), 5);
  assert.deepEqual(source.query_ids[question], expectedIDs);
  assert.deepEqual(fixed.maskIDs(referenceMask), expectedIDs);
  for (let team = 0; team < 6; team++) {
    const n = source.total_high[team] - expectedIDs.length;
    assert.equal(n, source.padding_counts[team]);
    assert(n >= 0 && n <= padding.known_negative_ids.length);
    result.padding_checks++;
  }
}
for (const cap of [1, 2]) {
  const states = referenceCandidates(cap);
  assert.deepEqual(fixed.candidates(cap).sort((a, b) => a - b), states);
  const signatures = new Set(), distribution = {}, maxima = Array(6).fill(0);
  const distinctPrefixes = Array.from({ length: 6 }, () => new Set());
  for (const truth of states) {
    const hits = referenceSignature(truth);
    assert.deepEqual(fixed.signature(truth), hits);
    signatures.add(hits.join(','));
    let firstUnique = 0;
    for (let depth = 0; depth <= 5; depth++) {
      const prefix = hits.slice(0, depth);
      const expected = referenceRemaining(cap, prefix);
      assert.deepEqual(fixed.remaining(cap, prefix).sort((a, b) => a - b), expected);
      assert(expected.includes(truth));
      maxima[depth] = Math.max(maxima[depth], expected.length);
      distinctPrefixes[depth].add(prefix.join(','));
      if (expected.length === 1 && !firstUnique) firstUnique = depth;
      if (depth < 5) {
        const expectedGroups = new Map();
        for (const member of expected) {
          const hit = referenceSignature(member)[depth];
          if (!expectedGroups.has(hit)) expectedGroups.set(hit, []);
          expectedGroups.get(hit).push(member);
        }
        const actualGroups = fixed.responseGroups(cap, prefix);
        assert.deepEqual([...actualGroups.keys()].sort(), [...expectedGroups.keys()].sort());
        for (const [hit, members] of expectedGroups) assert.deepEqual(actualGroups.get(hit).sort((a, b) => a - b), members);
        result.prefix_groups_checked++;
      } else assert.equal(fixed.responseGroups(cap, prefix).size, 0);
    }
    assert(firstUnique >= 1 && firstUnique <= 5);
    assert.deepEqual(fixed.remaining(cap, hits), [truth]);
    distribution[firstUnique] = (distribution[firstUnique] || 0) + 1;
    const trace = fixed.trace(cap, truth);
    assert.equal(trace.length, 5);
    trace.forEach((step, index) => {
      assert.equal(step.question, index + 1);
      assert.equal(step.query_mask, source.query_masks[index]);
      assert.deepEqual(step.selected_ids, source.query_ids[index]);
      assert.equal(step.hit, hits[index]);
      assert.equal(step.before, referenceRemaining(cap, hits.slice(0, index)).length);
      assert.equal(step.after, referenceRemaining(cap, hits.slice(0, index + 1)).length);
    });
    result.states_checked++;
  }
  assert.equal(signatures.size, cap === 1 ? 11 : 55);
  result.signature_counts[cap] = signatures.size;
  result.first_unique_depth_distribution[cap] = distribution;
  result.maximum_prefix_depth[cap] = Math.max(...Object.keys(distribution).map(Number));
  result.maximum_candidate_counts_by_prefix[cap] = maxima;
  result[`distinct_prefix_signatures_capacity_${cap}`] = distinctPrefixes.map(set => set.size);
}
assert.equal(result.states_checked, 66);
assert.deepEqual(result.maximum_prefix_depth, { 1: 5, 2: 5 });
assert.deepEqual(result.first_unique_depth_distribution, { 1: { 3: 5, 4: 4, 5: 2 }, 2: { 2: 1, 3: 5, 4: 19, 5: 30 } });
// A bitwise OR erases an answer of two and is not the operation being proved.
const pairORs = new Set(referenceCandidates(2).map(member => source.codewords.filter((_, i) => member & 2 ** i).reduce((sum, word) => sum | word, 0)));
assert(pairORs.size < 55);
result.pair_bitwise_or_signature_count = pairORs.size;
result.signature_operation = 'coordinatewise integer sum; each entry is 0, 1, or 2';

assert.throws(() => fixed.candidates(0));
assert.throws(() => fixed.popcount(-1));
assert.throws(() => fixed.maskIDs(2048));
assert.throws(() => fixed.signature(0));
assert.throws(() => fixed.signature(7));
assert.throws(() => fixed.signature(1, Array(6).fill(0)));
assert.throws(() => fixed.signature(1, [undefined]));
assert.throws(() => fixed.remaining(1, [2]));
assert.throws(() => fixed.remaining(2, [-1]));
assert.throws(() => fixed.remaining(2, [1.5]));
assert.throws(() => fixed.remaining(2, [0, 0, 0, 0, 0, 0]));
assert.throws(() => fixed.remaining(1, Array(1)));
assert.throws(() => fixed.trace(1, 3));
assert.deepEqual(fixed.remaining(1, [0, 0, 0, 0, 0]), []);
assert.deepEqual(fixed.signature(1, []), []);

// Exact independent ROC interpolation at FPR 1/100. Five selected candidate
// IDs plus known nonmembers keep each Team's total high count constant.
const gcd = (a, b) => { while (b) [a, b] = [b, a % b]; return a < 0n ? -a : a; };
const fraction = (n, d = 1n) => { n = BigInt(n); d = BigInt(d); assert(d !== 0n); const g = gcd(n, d); return { n: n / g, d: d / g }; };
const add = (a, b) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a, b) => fraction(a.n * b.n, a.d * b.d);
const divide = (a, b) => fraction(a.n * b.d, a.d * b.n);
const exact = a => a.d === 1n ? String(a.n) : `${a.n}/${a.d}`;
function meanFromROC(hits) {
  let sum = fraction(0);
  for (let team = 0; team < 6; team++) {
    const x = fraction(source.total_high[team] - hits[team], 24 - source.caps[team]);
    const y = fraction(hits[team], source.caps[team]);
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
  values.push(mean); result.score_patterns++;
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

const initial = source.caps.map(referenceCandidates);
function assertDisjoint(row, sets) {
  assert(row && row.length === 6);
  let used = 0;
  row.forEach((member, team) => { assert(sets[team].includes(member)); assert(!(member & used)); used |= member; });
  assert.equal(pc(used), 9);
}
function brute(sets, team = 0, used = 0) {
  if (team === 6) return true;
  return sets[team].some(member => !(member & used) && brute(sets, team + 1, used | member));
}
assertDisjoint(fixed.findDisjoint(initial), initial);
const truth = source.joint_demo_membership_masks;
assert.deepEqual(truth, [128, 3, 4, 24, 32, 320]);
const singletonSets = truth.map(member => [member]);
assertDisjoint(truth, singletonSets);
assert.deepEqual(fixed.findDisjoint(singletonSets), truth);
const conflict = singletonSets.map(set => set.slice()); conflict[2] = conflict[0].slice();
assert.equal(fixed.findDisjoint(conflict), null);
assert.equal(fixed.findDisjoint([[], ...initial.slice(1)]), null);
assert.throws(() => fixed.findDisjoint(initial.slice(1)));
assert.throws(() => fixed.findDisjoint([[0], ...initial.slice(1)]));
for (const code of Object.keys(encoding.integer_values).map(Number)) {
  const hits = tree.counts(code), sets = source.caps.map((cap, team) => fixed.remaining(cap, [hits[team]]));
  const found = fixed.findDisjoint(sets);
  if (found) { assertDisjoint(found, sets); result.initial_reachable_joint_patterns++; }
}
let seed = 20261009;
const random = n => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
for (let index = 0; index < 200; index++) {
  const sets = initial.map(options => Array.from({ length: 1 + random(4) }, () => options[random(options.length)]));
  const found = fixed.findDisjoint(sets);
  assert.equal(!!found, brute(sets));
  if (found) assertDisjoint(found, sets);
  result.disjoint_search_random_cases++;
}
result.joint_demo = [];
for (let question = 0; question < 5; question++) {
  const hits = truth.map(member => referenceSignature(member)[question]);
  const mean = meanFromROC(hits);
  const sets = source.caps.map((cap, team) => fixed.remaining(cap, referenceSignature(truth[team]).slice(0, question + 1)));
  assertDisjoint(fixed.findDisjoint(sets), sets);
  assert.equal(tree.decodeMean(exact(mean), encoding), hits.reduce((sum, hit, team) => sum + hit * 3 ** team, 0));
  result.joint_demo.push({ question: question + 1, responses: hits, mean_exact: exact(mean), remaining_by_team: sets.map(set => set.length) });
}
assert.deepEqual(source.caps.map((cap, team) => fixed.remaining(cap, referenceSignature(truth[team]))[0]), truth);
result.pedagogical_examples = {};
for (const cap of [1, 2]) {
  const member = source.demo_membership_masks[cap], steps = fixed.trace(cap, member);
  result.pedagogical_examples[cap] = { membership_mask: member, member_ids: fixed.maskIDs(member), responses: referenceSignature(member), remaining: [cap === 1 ? 11 : 55, ...steps.map(step => step.after)] };
  assert(steps[3].after > 1);
  assert.equal(steps[4].after, 1);
}
assert.deepEqual(result.pedagogical_examples[1].remaining, [11, 6, 3, 2, 2, 1]);
assert.deepEqual(result.pedagogical_examples[2].remaining, [55, 30, 15, 7, 2, 1]);
result.source_files = [sourcePath, encodingPath, paddingPath].map(file => ({ path: path.relative(root, file), sha256: sha256(fs.readFileSync(file)) }));
result.code_input_sha256 = sha256(JSON.stringify({ candidate_ids: source.candidate_ids, codewords: source.codewords, query_masks: source.query_masks }));
result.core_sha256 = sha256(fs.readFileSync(path.join(__dirname, 'fixed_core.js')));
if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_fixed_core.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
