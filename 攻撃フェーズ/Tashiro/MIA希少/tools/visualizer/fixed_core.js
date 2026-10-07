/* Fixed five-question construction. Answers never change the selected IDs. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MIA_FIXED = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const FULL = 2047;
  const IDS = Object.freeze([47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675]);
  const CODEWORDS = Object.freeze([26, 29, 14, 18, 9, 7, 8, 4, 17, 20, 3]);
  const QUERY_MASKS = Object.freeze([1330, 1069, 678, 87, 779]);
  const CAPS = Object.freeze([1, 2, 1, 2, 1, 2]);
  const HIGH = Object.freeze([6, 7, 7, 11, 10, 12]);
  const PADDING = Object.freeze([1, 2, 2, 6, 5, 7]);
  const POP = new Uint8Array(FULL + 1);
  for (let value = 1; value <= FULL; value++) POP[value] = POP[value >>> 1] + (value & 1);
  function mask(value) {
    if (!Number.isInteger(value) || value < 0 || value > FULL) throw new RangeError('Invalid 11-bit mask');
    return value;
  }
  function popcount(value) { return POP[mask(value)]; }
  function capacity(cap) {
    if (cap !== 1 && cap !== 2) throw new RangeError('Membership capacity must be one or two');
    return cap;
  }
  function membership(member) {
    mask(member); capacity(POP[member]); return member;
  }
  function maskIDs(value) {
    mask(value); return IDS.filter((_, index) => value & (1 << index));
  }
  function candidates(cap) {
    capacity(cap);
    const states = [];
    for (let i = 0; i < IDS.length; i++) {
      if (cap === 1) states.push(1 << i);
      else for (let j = i + 1; j < IDS.length; j++) states.push((1 << i) | (1 << j));
    }
    return states;
  }
  function signature(member, queries = QUERY_MASKS) {
    membership(member);
    if (!Array.isArray(queries) || queries.length > 5) throw new RangeError('At most five fixed questions are permitted');
    const hits = [];
    for (let index = 0; index < queries.length; index++) hits.push(POP[member & mask(queries[index])]);
    return hits;
  }
  function remaining(cap, hits) {
    capacity(cap);
    if (!Array.isArray(hits) || hits.length > 5) throw new RangeError('At most five fixed responses are permitted');
    for (let index = 0; index < hits.length; index++) {
      if (!Number.isInteger(hits[index]) || hits[index] < 0 || hits[index] > cap) throw new RangeError('Invalid membership count');
    }
    return candidates(cap).filter(member => {
      for (let index = 0; index < hits.length; index++) if (POP[member & QUERY_MASKS[index]] !== hits[index]) return false;
      return true;
    });
  }
  function groups(states, query) {
    mask(query);
    if (!Array.isArray(states)) throw new TypeError('Candidate masks must be an array');
    const result = new Map();
    for (const member of states) {
      membership(member);
      const count = POP[member & query];
      if (!result.has(count)) result.set(count, []);
      result.get(count).push(member);
    }
    return result;
  }
  function responseGroups(cap, hits) {
    const states = remaining(cap, hits);
    return hits.length === 5 ? new Map() : groups(states, QUERY_MASKS[hits.length]);
  }
  function trace(cap, member) {
    capacity(cap); membership(member);
    if (POP[member] !== cap) throw new RangeError('Membership has the wrong capacity');
    const hits = signature(member), history = [], steps = [];
    for (let index = 0; index < 5; index++) {
      const before = remaining(cap, history);
      history.push(hits[index]);
      const after = remaining(cap, history);
      steps.push({ question: index + 1, query_mask: QUERY_MASKS[index], selected_ids: maskIDs(QUERY_MASKS[index]), hit: hits[index], before: before.length, after: after.length, candidates: after });
    }
    return steps;
  }
  function findDisjoint(sets) {
    if (!Array.isArray(sets) || sets.length !== 6) throw new TypeError('Six Team candidate sets are required');
    const teams = sets.map((states, team) => {
      if (!Array.isArray(states)) throw new TypeError('Candidate masks must be arrays');
      const options = Array.from(new Set(states));
      options.forEach(member => { membership(member); if (POP[member] !== CAPS[team]) throw new RangeError('Membership has the wrong Team capacity'); });
      return { team, options };
    }).sort((a, b) => a.options.length - b.options.length || a.team - b.team);
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
        required += CAPS[teams[d].team];
      }
      if (POP[union] < required) { failed[key] = 1; return null; }
      const team = teams[depth];
      for (const member of team.options) {
        if (member & used) continue;
        row[team.team] = member;
        const found = visit(depth + 1, used | member);
        if (found) return found;
      }
      failed[key] = 1; return null;
    }
    return visit(0, 0);
  }
  return Object.freeze({ FULL, IDS, CODEWORDS, QUERY_MASKS, CAPS, HIGH, PADDING, popcount, maskIDs, candidates, signature, remaining, groups, responseGroups, trace, findDisjoint });
});
