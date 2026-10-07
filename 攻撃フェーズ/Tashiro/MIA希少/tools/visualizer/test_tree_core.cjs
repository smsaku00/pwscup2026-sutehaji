#!/usr/bin/env node
'use strict';
// Exhaustive replay against the committed proof, not a solver or UI fixture.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const core = require('./tree_core.js');
const proof = path.resolve(__dirname, '../../analysis/full_depth4_20261007');
const read = file => JSON.parse(fs.readFileSync(path.join(proof, file), 'utf8'));
const manifest = read('manifest.json');
const audit = read('independent_full_audit.json');
const encoding = read('universal_score_encoding.json');
const witness = read('example_witness_code30.json');
const pc = m => m.toString(2).replace(/0/g, '').length;
const powers = [1, 3, 9, 27, 81, 243];
const referenceResponse = (row, q) => row.reduce((sum, m, t) => sum + pc(m & q[t]) * powers[t], 0);
const compareRows = (a, b) => {
  for (let t = 0; t < 6; t++) if (a[t] !== b[t]) return a[t] - b[t];
  return 0;
};
function stateHash(rows) {
  const sorted = rows.slice().sort(compareRows), bytes = Buffer.alloc(sorted.length * 12);
  sorted.forEach((row, r) => row.forEach((m, t) => bytes.writeUInt16LE(m, r * 12 + t * 2)));
  return crypto.createHash('sha256').update(bytes).digest('hex');
}
const representativeCerts = new Map();
const representatives = new Map();
const entries = new Map();
for (const orbit of manifest.orbits) {
  const cert = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(proof, 'orbits', String(orbit.representative), 'certificate.json.gz'))));
  assert.equal(cert.q3_code, orbit.representative);
  assert.equal(cert.depth, 3);
  const rows = core.generate(orbit.representative);
  assert.equal(stateHash(rows), cert.state_hash);
  representativeCerts.set(orbit.representative, cert);
  representatives.set(orbit.representative, rows);
  for (const code of orbit.q3_codes) {
    assert(!entries.has(code));
    entries.set(code, { representative: orbit.representative, ...orbit.mappings[code] });
  }
}
assert.deepEqual(core.Q3, manifest.q3_query);
assert.deepEqual(core.CAPS, manifest.caps);
assert.deepEqual(core.IDS, manifest.candidate_ids);
const branchReports = new Map(audit.branches.map(b => [b.q3_code, b]));
const result = {
  status: 'passed', reachable_q3_branches: 0, empty_q3_branches: [], representative_state_hashes: representativeCerts.size,
  transported_state_sets: 0, branch_state_hashes: 0, total_states: 0, encoded_nodes_total: 1,
  q4_branches_total: 0, q4_max_bucket: 0, q5_cells_total: 0, q5_max_cell: 0,
  q6_nodes_total: 0, terminal_singleton_outcomes: 0, maximum_additional_depth: 1,
  query_padding_checks: 0, collisions: 0, decoded_mean_scores: 0, midpoint_rejections: 0,
};
function checkTree(rows, node, round) {
  if (node.leaf) { assert.equal(rows.length, 1); return rows.length; }
  assert(round <= 6);
  assert.equal(node.query.length, 6);
  node.query.forEach((m, t) => {
    assert.equal(core.popcount(m), pc(m));
    assert(pc(m) <= 5);
    const padding = encoding.total_high[t] - pc(m);
    assert(padding >= 1 && padding <= 13);
    result.query_padding_checks++;
  });
  result.encoded_nodes_total++;
  result.maximum_additional_depth = Math.max(result.maximum_additional_depth, round - 2);
  const groups = core.groups(rows, node.query);
  if (round === 4) {
    result.q4_branches_total += groups.size;
    result.q4_max_bucket = Math.max(result.q4_max_bucket, ...Array.from(groups.values(), g => g.length));
  }
  if (round === 5) {
    result.q5_cells_total += groups.size;
    result.q5_max_cell = Math.max(result.q5_max_cell, ...Array.from(groups.values(), g => g.length));
  }
  for (const [code, group] of groups) {
    // Recompute counts separately from the lookup-table implementation.
    assert.equal(referenceResponse(group[0], node.query), code);
    assert.deepEqual(core.counts(code), powers.map(p => Math.floor(code / p) % 3));
  }
  if (node.terminal) {
    assert.equal(round, 6);
    assert(!node.children);
    result.q6_nodes_total++;
    for (const group of groups.values()) assert.equal(group.length, 1, 'Terminal collision');
    result.terminal_singleton_outcomes += groups.size;
    return rows.length;
  }
  assert.deepEqual(Object.keys(node.children).map(Number).sort((a, b) => a - b), Array.from(groups.keys()).sort((a, b) => a - b));
  let count = 0;
  for (const [code, group] of groups) count += checkTree(group, node.children[code], round + 1);
  assert.equal(count, rows.length);
  return count;
}
function referenceTransportRow(row, pm, tm) {
  const out = Array(6).fill(0);
  for (let t = 0; t < 6; t++) for (let i = 0; i < 11; i++) {
    if (row[t] & (1 << i)) out[tm[t]] |= 1 << pm[i];
  }
  return out;
}
for (const code of Object.keys(encoding.integer_values).map(Number).sort((a, b) => a - b)) {
  const rows = core.generate(code);
  if (!rows.length) {
    assert(!entries.has(code));
    result.empty_q3_branches.push(code);
    continue;
  }
  const mapping = entries.get(code);
  assert(mapping, 'Missing q3 orbit');
  const pm = mapping.candidate_old_to_new, tm = mapping.team_old_to_new;
  assert.deepEqual([...pm].sort((a, b) => a - b), Array.from({ length: 11 }, (_, i) => i));
  assert.deepEqual([...tm].sort((a, b) => a - b), Array.from({ length: 6 }, (_, i) => i));
  for (let t = 0; t < 6; t++) for (let i = 0; i < 11; i++) {
    assert.equal(core.CAPS[t], core.CAPS[tm[t]]);
    assert.equal((core.Q3[t] >>> i) & 1, (core.Q3[tm[t]] >>> pm[i]) & 1);
  }
  const movedRows = representatives.get(mapping.representative).map(row => referenceTransportRow(row, pm, tm));
  assert.equal(movedRows.length, rows.length);
  const actualSet = new Set(rows.map(r => r.join(',')));
  assert.equal(actualSet.size, rows.length);
  for (const row of movedRows) assert(actualSet.has(row.join(',')));
  for (const row of rows) {
    assert.equal(referenceResponse(row, core.Q3), code);
    assert.equal(core.response(row, core.Q3), code);
    assert.equal(row.reduce((a, m) => a | m, 0).toString(2).replace(/0/g, '').length, 9);
    row.forEach((m, t) => assert.equal(pc(m), core.CAPS[t]));
  }
  const branch = branchReports.get(code);
  assert.equal(rows.length, branch.states);
  assert.equal(stateHash(rows), branch.state_hash);
  const tree = core.normalize(core.moveTree(representativeCerts.get(mapping.representative).tree, pm, tm));
  assert.equal(checkTree(rows, tree, 4), rows.length);
  result.reachable_q3_branches++;
  result.transported_state_sets++;
  result.branch_state_hashes++;
  result.total_states += rows.length;
}
for (const key of ['total_states', 'reachable_q3_branches', 'q4_branches_total', 'q4_max_bucket', 'q5_cells_total', 'q5_max_cell', 'encoded_nodes_total']) {
  assert.equal(result[key], audit[key], key);
}
assert.equal(result.q6_nodes_total, audit.q6_nodes_total);
assert.equal(result.maximum_additional_depth, 4);
assert.equal(result.terminal_singleton_outcomes, 2494800);
assert.deepEqual(result.empty_q3_branches, [...audit.unreachable_q3_codes].sort((a, b) => a - b));

// Reproduce all 20 outcomes of the documented normalized example.
let witnessNode = core.normalize(representativeCerts.get(30).tree);
let witnessRows = core.generate(30);
assert.deepEqual(witnessNode.query, witness.query_masks_normalized[1]);
witnessRows = core.groups(witnessRows, witnessNode.query).get(witness.q4_code);
assert.equal(witnessRows.length, 631);
witnessNode = witnessNode.children[witness.q4_code];
assert.deepEqual(witnessNode.query, witness.query_masks_normalized[2]);
witnessRows = core.groups(witnessRows, witnessNode.query).get(witness.q5_code);
assert.equal(witnessRows.length, 20);
witnessNode = witnessNode.children[witness.q5_code];
assert(witnessNode.terminal);
assert.deepEqual(witnessNode.query, witness.query_masks_normalized[3]);
assert.deepEqual(witnessRows.slice().sort(compareRows), witness.q5_cell_states.slice().sort(compareRows));
const witnessOutcomes = core.groups(witnessRows, witnessNode.query);
assert.deepEqual([...witnessOutcomes.keys()].sort((a, b) => a - b), [...witness.q6_response_codes].sort((a, b) => a - b));
assert.equal(witnessOutcomes.get(328).length, 1);
result.witness_states = 20;
result.witness_history = [30, 253, 246, 328];

// Test the complete score alphabet and reject ambiguous or unsupported input.
const alphabet = Object.entries(encoding.integer_values).map(([code, v]) => ({ code: Number(code), v })).sort((a, b) => a.v - b.v);
for (const { code, v } of alphabet) {
  assert.equal(core.decodeMean(`${v}/${encoding.mean_denominator}`, encoding), code);
  assert.equal(core.decodeMean(v / encoding.mean_denominator, encoding), code);
  assert.equal(core.decodeMean((v / encoding.mean_denominator).toFixed(6), encoding), code);
  const hs = core.counts(code);
  const fromFormula = hs.reduce((sum, h, t) => sum + (24 - core.CAPS[t]) * h / (100 * core.CAPS[t] * (encoding.total_high[t] - h)), 0) / 6;
  assert(Math.abs(fromFormula - v / encoding.mean_denominator) < 1e-16);
  result.decoded_mean_scores++;
}
let smallestGap = Infinity;
for (let i = 1; i < alphabet.length; i++) {
  const left = alphabet[i - 1].v, right = alphabet[i].v;
  const midExact = `${left + right}/${encoding.mean_denominator * 2}`;
  assert.throws(() => core.decodeMean(midExact, encoding));
  assert.throws(() => core.decodeMean((left + right) / (encoding.mean_denominator * 2), encoding));
  result.midpoint_rejections++;
  smallestGap = Math.min(smallestGap, (right - left) / encoding.mean_denominator);
}
assert.equal(smallestGap, encoding.minimum_gap);
result.minimum_mean_gap_exact = encoding.minimum_gap_exact;
for (const code of [-1, 2, 729, 1.5, NaN, Infinity, '30']) {
  assert.deepEqual(core.generate(code), []);
  assert.throws(() => core.counts(code));
}
for (const value of ['', '0x10', NaN, Infinity, '-1', '1', '1/0', 'bad']) assert.throws(() => core.decodeMean(value, encoding));
assert.throws(() => core.response([0, 0, 0, 0, 0, 0], [2048, 0, 0, 0, 0, 0]));
assert.throws(() => core.moveTree({ leaf: true }, Array(11).fill(0), [0, 1, 2, 3, 4, 5]));
assert.deepEqual(core.maskIDs(57), [47, 115, 169, 264]);
assert.equal(core.normalize({ query: [2047, 0, 0, 0, 0, 0], terminal: true }).query[0], 0);
assert.deepEqual(core.normalize({ leaf: true }), { leaf: true });
result.source_files = ['manifest.json', 'independent_full_audit.json', 'universal_score_encoding.json', 'example_witness_code30.json', 'orbits/*/certificate.json.gz'];
result.core_sha256 = crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname, 'tree_core.js'))).digest('hex');
if (process.argv.includes('--write-report')) fs.writeFileSync(path.join(__dirname, 'validation_core.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
