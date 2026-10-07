/* Read-only navigation of the saved independent Team policies. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MIA_FIVE = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const FULL = 2047, CAPS = Object.freeze([1, 2, 1, 2, 1, 2]);
  const POP = new Uint8Array(FULL + 1);
  for (let m = 1; m <= FULL; m++) POP[m] = POP[m >>> 1] + (m & 1);
  function mask(value) {
    if (!Number.isInteger(value) || value < 0 || value > FULL) throw new RangeError('Invalid 11-bit mask');
    return value;
  }
  function popcount(value) { return POP[mask(value)]; }
  function candidates(cap) {
    if (cap !== 1 && cap !== 2) throw new RangeError('Team capacity must be one or two');
    const out = [];
    for (let i = 0; i < 11; i++) {
      if (cap === 1) out.push(1 << i);
      else for (let j = i + 1; j < 11; j++) out.push((1 << i) | (1 << j));
    }
    return out;
  }
  function complete(node) { return !!node && Object.prototype.hasOwnProperty.call(node, 'member_mask'); }
  function query(node) {
    if (!node || typeof node !== 'object') throw new TypeError('Missing saved policy node');
    if (complete(node)) return 0;
    const q = mask(node.query_mask);
    if (POP[q] > 5) throw new RangeError('Saved question is not normalized');
    return q;
  }
  function groups(states, q) {
    mask(q);
    if (!Array.isArray(states)) throw new TypeError('Candidate masks must be an array');
    const out = new Map();
    for (const member of states) {
      mask(member);
      const h = POP[member & q];
      if (!out.has(h)) out.set(h, []);
      out.get(h).push(member);
    }
    return out;
  }
  function navigateTeam(policy, cap, hits) {
    let states = candidates(cap);
    if (!Array.isArray(hits) || hits.length > 5) throw new RangeError('At most five Team responses are permitted');
    const policies = policy && (policy.policies || policy);
    let node = policies && (policies[String(cap)] || (Object.prototype.hasOwnProperty.call(policies, 'query_mask') || complete(policies) ? policies : null));
    if (!node) throw new TypeError('Missing saved Team policy');
    function checkNode() {
      if (complete(node)) {
        mask(node.member_mask);
        if (states.length !== 1 || states[0] !== node.member_mask || POP[node.member_mask] !== cap) throw new Error('Saved leaf disagrees with candidate membership');
      } else {
        query(node);
        if (!node.children || node.before !== states.length) throw new Error('Saved question disagrees with candidate count');
      }
    }
    checkNode();
    for (const h of hits) {
      if (!Number.isInteger(h) || h < 0 || h > cap) throw new RangeError('Invalid Team response');
      if (complete(node)) {
        if (h !== 0) throw new RangeError('A completed Team has an empty question and must answer zero');
        continue;
      }
      const next = groups(states, query(node)).get(h);
      if (!next || !Object.prototype.hasOwnProperty.call(node.children, String(h))) throw new RangeError('Unreachable Team response');
      states = next; node = node.children[String(h)]; checkNode();
    }
    return { node, candidates: states, complete: complete(node), query: query(node) };
  }
  function findDisjoint(sets) {
    if (!Array.isArray(sets) || sets.length !== 6) throw new TypeError('Six Team candidate sets are required');
    const teams = sets.map((states, t) => {
      if (!Array.isArray(states)) throw new TypeError('Candidate masks must be arrays');
      const options = Array.from(new Set(states));
      options.forEach(m => { mask(m); if (POP[m] !== CAPS[t]) throw new RangeError('Candidate membership has the wrong Team capacity'); });
      return { t, options };
    }).sort((a, b) => a.options.length - b.options.length || a.t - b.t);
    if (teams.some(team => !team.options.length)) return null;
    const failed = new Uint8Array(7 * (FULL + 1)), row = Array(6);
    function visit(depth, used) {
      if (depth === 6) return row.slice();
      const key = depth * (FULL + 1) + used;
      if (failed[key]) return null;
      let union = 0, required = 0;
      for (let d = depth; d < 6; d++) {
        let available = false;
        for (const member of teams[d].options) if (!(member & used)) { union |= member; available = true; }
        if (!available) { failed[key] = 1; return null; }
        required += CAPS[teams[d].t];
      }
      if (POP[union] < required) { failed[key] = 1; return null; }
      const team = teams[depth];
      for (const member of team.options) {
        if (member & used) continue;
        row[team.t] = member;
        const found = visit(depth + 1, used | member);
        if (found) return found;
      }
      failed[key] = 1;
      return null;
    }
    return visit(0, 0);
  }
  return Object.freeze({ FULL, CAPS, popcount, candidates, complete, query, groups, navigateTeam, findDisjoint });
});
