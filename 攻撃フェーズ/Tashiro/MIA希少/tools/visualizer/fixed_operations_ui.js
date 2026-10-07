(function () {
  'use strict';
  const el = id => document.getElementById(id);
  const teams = data.teams, caps = data.caps;
  let observations = Array(5).fill(null), meanInputs = Array(5).fill(null);
  const names = mask => shared.maskIDs(mask, ids).join('・');
  const hitsCode = hits => hits.reduce((code, hit, t) => code + hit * 3 ** t, 0);
  function meanExact(hits) {
    let n = BigInt(data.encoding.integer_values[String(hitsCode(hits))]);
    let d = BigInt(data.encoding.mean_denominator);
    const gcd = (a, b) => b ? gcd(b, a % b) : a;
    const g = gcd(n, d);
    return `${n / g}/${d / g}`;
  }
  function candidatesFor(next) {
    return caps.map((cap, t) => five.candidates(cap).filter(member => {
      const signature = five.signature(member);
      return next.every((hits, r) => hits === null || signature[r] === hits[t]);
    }));
  }
  function accept(next, values) {
    if (!Array.isArray(next) || next.length !== 5) throw Error('5回の回答を指定してください。');
    next.forEach(hits => {
      if (hits !== null && (!Array.isArray(hits) || hits.length !== 6 || hits.some((h, t) => !Number.isInteger(h) || h < 0 || h > caps[t]))) throw Error('Teamの人数に合う回答ではありません。');
    });
    const sets = candidatesFor(next);
    if (!five.findDisjoint(sets)) throw Error('この回答では所属配置が成立しません。固定質問・点数・入力した桁を確認してください。');
    observations = next.map(hits => hits === null ? null : hits.slice());
    meanInputs = values.slice();
    render();
  }
  function run(action) {
    el('fixed-error').hidden = true;
    try {action();} catch (error) {el('fixed-error').textContent = error.message || String(error); el('fixed-error').hidden = false;}
  }
  function decodeRow(round) {
    let code;
    const value = el(`fixed-mean-${round + 1}`).value.trim();
    try {code = shared.decodeMean(value, data.encoding);} catch (error) {throw Error('平均点を読み取れません。生の数値の全桁を確認してください。境界や対応表の範囲外の値では進めません。');}
    const next = observations.map(hits => hits === null ? null : hits.slice()), values = meanInputs.slice();
    next[round] = shared.counts(code);
    values[round] = value;
    accept(next, values);
  }
  function render() {
    const sets = candidatesFor(observations), complete = sets.every(set => set.length === 1);
    const filled = observations.filter(hits => hits !== null).length;
    el('fixed-status').textContent = `${filled}/5回の平均点を照合済み。質問1〜5の調査IDはすべて固定です。`;
    observations.forEach((hits, r) => {el(`fixed-answer-${r + 1}`).textContent = hits ? teams.map((team, t) => `${team}:${hits[t]}人`).join(' ／ ') : '未入力';});
    el('fixed-results').innerHTML = '<table><thead><tr><th>Team</th><th>所属人数</th><th>残る候補数</th><th>候補ID</th></tr></thead><tbody>' + sets.map((set, t) => `<tr><td>${teams[t]}</td><td>${caps[t]}人</td><td>${set.length}${caps[t] === 2 ? '組' : '候補'}</td><td class="candidate-list">${set.map(names).join(' ／ ')}</td></tr>`).join('') + '</tbody></table>' + (complete ? `<div class="fixed-result"><b>全6 Teamの所属が1候補ずつに定まりました。</b><p class="small">入力した値が実際の公式平均点なら、この所属配置に確定します。説明用ボタンで入れた値は仮定例です。</p><p class="small">対象6 Team外：${names(2047 ^ sets.reduce((union, set) => union | set[0], 0))}</p></div>` : '');
  }
  for (let r = 0; r < 5; r++) el(`fixed-decode-${r + 1}`).onclick = () => run(() => decodeRow(r));
  el('fixed-demo').onclick = () => run(() => {
    const next = data.query_masks.map((query, r) => data.demo_masks.map(member => five.signature(member)[r]));
    const values = next.map(hits => String(data.encoding.integer_values[String(hitsCode(hits))] / data.encoding.mean_denominator));
    accept(next, values);
    values.forEach((value, r) => {el(`fixed-mean-${r + 1}`).value = value;});
  });
  el('fixed-clear').onclick = () => run(() => {
    accept(Array(5).fill(null), Array(5).fill(null));
    for (let r = 0; r < 5; r++) el(`fixed-mean-${r + 1}`).value = '';
  });
  el('fixed-export').onclick = () => {
    const sets = candidatesFor(observations);
    const body = JSON.stringify({strategy: 'nonadaptive', team_order: teams, fixed_query_masks: data.query_masks,
      query_ids: data.query_masks.map(query => shared.maskIDs(query, ids)), mean_values: meanInputs,
      responses: observations, remaining_counts: sets.map(set => set.length),
      members_by_team: sets.every(set => set.length === 1) ? Object.fromEntries(sets.map((set, t) => [teams[t], shared.maskIDs(set[0], ids)])) : null}, null, 2);
    const url = URL.createObjectURL(new Blob([body], {type: 'application/json'})), a = document.createElement('a');
    a.href = url; a.download = 'MIA_nonadaptive5_result.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  globalThis.MIA_FIXED_VIEWER = Object.freeze({decodeRow, meanExact, candidatesFor, accept, render,
    snapshot: () => ({observations: observations.map(hits => hits === null ? null : hits.slice()),
      meanInputs: meanInputs.slice(), counts: candidatesFor(observations).map(set => set.length),
      queries: data.query_masks.slice()})});
  render();
})();
