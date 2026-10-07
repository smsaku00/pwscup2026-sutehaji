/* Pure navigation of the certified tree. No network, DOM, or submission writes. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.MIA_TREE = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const CAPS = Object.freeze([1, 2, 1, 2, 1, 2]);
  const Q3 = Object.freeze([590, 436, 1547, 440, 1095, 433]);
  const IDS = Object.freeze([47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675]);
  const FULL = 2047;
  const POW3 = [1, 3, 9, 27, 81, 243];
  const POP = new Uint8Array(FULL + 1);
  for (let m = 1; m <= FULL; m++) POP[m] = POP[m >>> 1] + (m & 1);
  const OPTIONS = CAPS.map((cap, t) => {
    const buckets = Array.from({ length: cap + 1 }, () => []);
    for (let m = 0; m <= FULL; m++) if (POP[m] === cap) buckets[POP[m & Q3[t]]].push(m);
    return buckets;
  });

  function legalCode(code) {
    if (!Number.isInteger(code) || code < 0 || code >= 729) return false;
    return CAPS.every((cap, t) => Math.floor(code / POW3[t]) % 3 <= cap);
  }
  function counts(code) {
    if (!legalCode(code)) throw new RangeError('Invalid capacity-legal answer code');
    return POW3.map(p => Math.floor(code / p) % 3);
  }
  function validMask(mask) {
    if (!Number.isInteger(mask) || mask < 0 || mask > FULL) throw new RangeError('Invalid 11-bit mask');
    return mask;
  }
  function popcount(mask) { return POP[validMask(mask)]; }
  function validateQuery(q) {
    if (!Array.isArray(q) || q.length !== 6) throw new TypeError('A query must contain six masks');
    q.forEach(validMask);
    return q;
  }
  function generate(code) {
    if (!legalCode(code)) return [];
    const hs = counts(code), rows = [], row = Array(6);
    function visit(t, used) {
      if (t === 6) { rows.push(row.slice()); return; }
      for (const m of OPTIONS[t][hs[t]]) {
        if (!(m & used)) { row[t] = m; visit(t + 1, used | m); }
      }
    }
    visit(0, 0);
    return rows;
  }
  function responseUnchecked(row, q) {
    let code = 0;
    for (let t = 0; t < 6; t++) code += POP[row[t] & q[t]] * POW3[t];
    return code;
  }
  function response(row, q) {
    validateQuery(q);
    if (!Array.isArray(row) || row.length !== 6) throw new TypeError('A state must contain six masks');
    row.forEach(validMask);
    return responseUnchecked(row, q);
  }
  function groups(rows, q) {
    validateQuery(q);
    const out = new Map();
    for (const row of rows) {
      const code = responseUnchecked(row, q);
      if (!out.has(code)) out.set(code, []);
      out.get(code).push(row);
    }
    return out;
  }
  function permutation(p, length) {
    if (!Array.isArray(p) || p.length !== length || new Set(p).size !== length ||
        p.some(i => !Number.isInteger(i) || i < 0 || i >= length)) {
      throw new RangeError('Invalid transport permutation');
    }
  }
  function moveTree(node, pm, tm) {
    permutation(pm, 11); permutation(tm, 6);
    if (CAPS.some((cap, t) => cap !== CAPS[tm[t]])) throw new RangeError('Transport changes team capacities');
    const lut = new Uint16Array(FULL + 1);
    for (let m = 1; m <= FULL; m++) {
      let moved = 0;
      for (let i = 0; i < 11; i++) if (m & (1 << i)) moved |= 1 << pm[i];
      lut[m] = moved;
    }
    function move(n) {
      if (n.leaf) return { leaf: true };
      validateQuery(n.query);
      const q = Array(6);
      for (let t = 0; t < 6; t++) q[tm[t]] = lut[n.query[t]];
      if (n.terminal) return { query: q, terminal: true };
      if (!n.children || typeof n.children !== 'object') throw new TypeError('Missing certified children');
      const children = {};
      for (const [code, child] of Object.entries(n.children)) {
        const hs = counts(Number(code));
        let movedCode = 0;
        for (let t = 0; t < 6; t++) movedCode += hs[t] * POW3[tm[t]];
        children[movedCode] = move(child);
      }
      return { query: q, children };
    }
    return move(node);
  }
  function normalize(node) {
    if (node.leaf) return { leaf: true };
    validateQuery(node.query);
    const flips = node.query.map(m => POP[m] > 5);
    const q = node.query.map((m, t) => flips[t] ? FULL ^ m : m);
    if (node.terminal) return { query: q, terminal: true };
    if (!node.children || typeof node.children !== 'object') throw new TypeError('Missing certified children');
    const children = {};
    for (const [code, child] of Object.entries(node.children)) {
      const hs = counts(Number(code));
      const normalizedCode = hs.reduce((sum, h, t) => sum + (flips[t] ? CAPS[t] - h : h) * POW3[t], 0);
      if (Object.prototype.hasOwnProperty.call(children, normalizedCode)) throw new Error('Normalization child collision');
      children[normalizedCode] = normalize(child);
    }
    return { query: q, children };
  }
  function maskIDs(mask, ids = IDS) {
    validMask(mask);
    if (!Array.isArray(ids) || ids.length !== 11) throw new TypeError('Eleven candidate labels required');
    return ids.filter((id, i) => (mask & (1 << i)) !== 0);
  }

  // Decimal input and the proof's integer alphabet share an exact denominator.
  // A tiny double tolerance rejects rounded representations of a midpoint too.
  function rational(value) {
    if (typeof value !== 'string' && typeof value !== 'number') throw new TypeError('Invalid score');
    if (typeof value === 'number' && !Number.isFinite(value)) throw new RangeError('Score must be finite');
    const s = String(value).trim();
    const fraction = /^([+-]?\d+)\/(\d+)$/.exec(s);
    if (fraction) {
      const d = BigInt(fraction[2]);
      if (!d) throw new RangeError('Score denominator cannot be zero');
      return { n: BigInt(fraction[1]), d };
    }
    const decimal = /^([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:[eE]([+-]?\d+))?$/.exec(s);
    if (!decimal) throw new TypeError('Invalid decimal score');
    const places = (decimal[3] || decimal[4] || '').length;
    const exponent = Number(decimal[5] || 0) - places;
    if (!Number.isSafeInteger(exponent) || Math.abs(exponent) > 1000) throw new RangeError('Score exponent is out of range');
    let n = BigInt((decimal[2] || '0') + (decimal[3] || decimal[4] || ''));
    if (decimal[1] === '-') n = -n;
    return exponent >= 0 ? { n: n * 10n ** BigInt(exponent), d: 1n } : { n, d: 10n ** BigInt(-exponent) };
  }
  const decoderCache = new WeakMap();
  function decodeMean(value, encoding) {
    const score = rational(value);
    let alphabet = decoderCache.get(encoding);
    if (!alphabet) {
      if (!Number.isSafeInteger(encoding.mean_denominator) || encoding.mean_denominator <= 0) throw new Error('Invalid mean denominator');
      const denominator = BigInt(encoding.mean_denominator);
      const gap = rational(encoding.minimum_gap_exact);
      if (gap.n <= 0n) throw new Error('Invalid decoder margin');
      const entries = Object.entries(encoding.integer_values).map(([code, v]) => {
        if (!legalCode(Number(code)) || !Number.isSafeInteger(v)) throw new Error('Invalid decoder alphabet');
        return { code: Number(code), integer: BigInt(v) };
      });
      if (entries.length !== 216) throw new Error('Incomplete decoder alphabet');
      alphabet = { denominator, gap, entries };
      decoderCache.set(encoding, alphabet);
    }
    const ranked = alphabet.entries.map(entry => {
      const delta = score.n * alphabet.denominator - entry.integer * score.d;
      return { code: entry.code, distance: delta < 0n ? -delta : delta };
    }).sort((a, b) => a.distance < b.distance ? -1 : a.distance > b.distance ? 1 : 0);
    const nearest = ranked[0], denominator = score.d * alphabet.denominator;
    const limit = alphabet.gap.n * denominator;
    if (nearest.distance * 2n * alphabet.gap.d >= limit) throw new RangeError('Score exceeds half of the certified minimum gap');
    const numericDistance = Number(nearest.distance) / Number(denominator);
    const runnerDistance = Number(ranked[1].distance) / Number(denominator);
    const halfGap = Number(alphabet.gap.n) / Number(alphabet.gap.d) / 2;
    const scoreNumber = Number(score.n) / Number(score.d);
    const tolerance = Number.EPSILON * Math.max(1, Math.abs(scoreNumber)) * 8;
    if (Math.abs(numericDistance - halfGap) <= tolerance || Math.abs(runnerDistance - numericDistance) <= tolerance) {
      throw new RangeError('Ambiguous rounded score at a decoder boundary');
    }
    return nearest.code;
  }

  return Object.freeze({ CAPS, Q3, IDS, FULL, counts, popcount, generate, response, groups, moveTree, normalize, maskIDs, decodeMean });
});
