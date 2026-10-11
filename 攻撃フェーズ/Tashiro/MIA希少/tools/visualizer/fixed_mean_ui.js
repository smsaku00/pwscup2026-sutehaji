/* Finite hypotheses are scored forward, then compared with one returned mean. */
(function () {
  'use strict';
  const el = id => document.getElementById(id), targetHits = [0, 1, 0, 1, 1, 1];
  let stage = 0, assumed = Array(6).fill(0);
  const gcd = (a, b) => b ? gcd(b, a % b) : a;
  function fraction(n, d = 1n) {n = BigInt(n); d = BigInt(d); const g = gcd(n, d); return {n: n / g, d: d / g};}
  const add = (a, b) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
  const text = a => `${a.n}/${a.d}`, number = a => Number(a.n) / Number(a.d);
  const decimal = n => Number(n.toPrecision(12)).toString();
  const score = (n, m, h) => fraction(h * (24 - m), 100 * m * (n - h));
  const sum = values => values.reduce(add, fraction(0));
  function prediction(hits) {
    const scores = hits.map((h, i) => score(data.high[i], data.caps[i], h));
    const total = sum(scores), average = fraction(total.n, total.d * 6n);
    return {hits: hits.slice(), contributions: scores.map((value, i) => ({team: data.teams[i], members: data.caps[i], high: data.high[i], hits: hits[i], exact: text(value), value: number(value)})), sumExact: text(total), meanExact: text(average), meanValue: number(average)};
  }
  const target = prediction(targetHits);
  const candidates = [];
  function enumerate(prefix) {
    if (prefix.length === 6) {candidates.push(prediction(prefix)); return;}
    for (let h = 0; h <= data.caps[prefix.length]; h++) enumerate([...prefix, h]);
  }
  enumerate([]);
  const matches = candidates.filter(row => row.meanExact === target.meanExact);
  if (candidates.length !== 216 || new Set(candidates.map(row => row.meanExact)).size !== 216 || matches.length !== 1) throw Error('教材の平均点対応を確認してください。');
  function model() {
    const current = prediction(assumed);
    return {...current, stage, targetExact: target.meanExact, targetValue: target.meanValue, matchesTarget: current.meanExact === target.meanExact, candidateCount: candidates.length, matchCount: matches.length, inferred: stage === 3 ? matches[0].hits.slice() : null};
  }
  function historyPrediction(row, round) {
    const c = data.intro.conditions, sizes = c[round === 1 ? 'round1_sizes' : 'round2_sizes'];
    const active = c.teams.map((team, i) => ({team, members: row.counts[team], high: sizes[i]})).filter(x => x.members);
    const scores = active.map(x => score(x.high, x.members, x.members)), total = sum(scores);
    const average = active.length ? fraction(total.n, total.d * BigInt(active.length)) : null;
    return {contributions: active.map((x, i) => ({...x, exact: text(scores[i]), value: number(scores[i])})), denominator: active.length, excluded: c.teams.length - active.length, sumExact: text(total), meanExact: average ? text(average) : null, meanValue: average ? number(average) : null};
  }
  function candidatePlot() {
    const sorted = candidates.slice().sort((a, b) => a.meanValue - b.meanValue), max = sorted[sorted.length - 1].meanValue;
    const px = value => 28 + value / max * 624;
    return `<svg viewBox="0 0 680 78" role="img" aria-label="216組の予測平均はすべて異なり、教材の返却点に一致するのは1組"><path d="M28,46 H652" class="roc-axis"/>${sorted.map((row, i) => `<circle data-mean-hits="${row.hits.join('')}" data-mean-exact="${row.meanExact}" cx="${px(row.meanValue)}" cy="${18 + i % 3 * 9}" r="${row.meanExact === target.meanExact ? 4 : 2}" class="mean-candidate${row.meanExact === target.meanExact ? ' matched' : ' excluded'}"><title>${row.hits.join(' ')} → ${row.meanExact}</title></circle>`).join('')}<path d="M${px(target.meanValue)},10 V49" class="history-observed"/><text x="340" y="67" text-anchor="middle">216組 → 215組が不一致 → 1組が一致</text></svg>`;
  }
  function render() {
    const state = model();
    el('intro-page-bridge').setAttribute('data-mean-stage', String(stage));
    el('bridge-returned').innerHTML = `<span>教材で返った平均点（1つ）</span><b>${state.targetExact}</b><small> ≈ ${decimal(state.targetValue)}</small><span>${stage === 3 ? '一致する組が1つなので、各チームのTPRを読み取れます。' : '各チームのTPRは、まだ個別には見えません。'}</span>`;
    data.teams.forEach((team, i) => {
      el(`bridge-team-${i}`).textContent = `Team ${team}`;
      el(`bridge-known-${i}`).textContent = `既知：所属 m＝${data.caps[i]}人 ／ 高側 n＝${data.high[i]}人`;
      const input = el(`bridge-hits-${i}`);
      input.setAttribute('aria-label', `Team ${team} の高側所属人数 h を仮置き`);
      input.innerHTML = Array.from({length: data.caps[i] + 1}, (_, h) => `<option value="${h}"${assumed[i] === h ? ' selected' : ''}>${h}人</option>`).join(''); input.value = String(assumed[i]);
      el(`bridge-choices-${i}`).textContent = Array.from({length: data.caps[i] + 1}, (_, h) => `${h}人 → ${text(score(data.high[i], data.caps[i], h))}`).join(' ／ ');
      el(`bridge-tpr-${i}`).innerHTML = `<span>${stage === 3 ? '一致した組のTPR' : '仮定から計算したTPR'}</span><b>${state.contributions[i].exact}</b><small> ≈ ${decimal(state.contributions[i].value)}</small>`;
    });
    el('bridge-assumption-label').textContent = stage === 3 ? '④ 今回の平均点に一致した高側人数hの組' : '① 所属人数mは固定：今回の高側人数hだけを仮置き';
    el('bridge-calculate').disabled = stage >= 1; el('bridge-compare').disabled = stage !== 1; el('bridge-compare-all').disabled = stage !== 2;
    el('bridge-example').innerHTML = stage === 0
      ? '<p>第1・2回で分かったmは固定し、6チームの高側人数hを仮置きしてください。各TPRを計算し、この6つを足して6で割ります。今回は平均の対象チームも分母も変わりません。</p>'
      : `<div class="mean-addends">${state.contributions.map(x => `<span><small>Team ${x.team}</small><b>${x.exact}</b></span>`).join('<em>＋</em>')}</div><div class="mean-equation"><span>② 合計 <b>${state.sumExact}</b></span><span> ÷ <b>6チームで固定</b><small>h＝0のチームも含む</small></span><span>＝ 今回の予測平均 <b>${state.meanExact}</b><small> ≈ ${decimal(state.meanValue)}</small></span></div>`;
    el('bridge-comparison').hidden = stage < 2;
    el('bridge-comparison').innerHTML = stage < 2 ? '' : `<div class="mean-verdict ${state.matchesTarget ? 'matched' : 'mismatch'}"><b>③ 予測平均 ${state.meanExact} ${state.matchesTarget ? '＝' : '≠'} 返却点 ${state.targetExact}</b><span>${stage === 3 ? '216組を計算して照合すると、この1組だけが一致。各チームのTPRと高側人数 h が分かります。' : state.matchesTarget ? 'この組は一致。ただし、他の組も同じ平均になるか確認してから特定します。' : 'この組は不一致。仮置きした人数の組を候補から外します。'}</span></div>` + (stage === 3 ? candidatePlot() : '');
    el('bridge-inferred').hidden = stage !== 3;
    el('bridge-inferred').textContent = stage === 3 ? `Team ${data.teams.join('・')} の順：${state.inferred.join('　')} → 提出1の横6桁として使える` : '';
  }
  function refresh() {if (globalThis.MIA_FIXED_INTRO) globalThis.MIA_FIXED_INTRO.render(); else render();}
  function setHits(values) {
    if (!Array.isArray(values) || values.length !== 6 || values.some((h, i) => !Number.isInteger(h) || h < 0 || h > data.caps[i])) throw Error('各チームの高側人数を確認してください。');
    assumed = values.slice(); stage = 0; refresh();
  }
  function setStage(value) {
    if (![0, 1, 2, 3].includes(value)) throw Error('平均点の照合段階を確認してください。');
    stage = value; if (stage === 3) assumed = matches[0].hits.slice(); refresh();
  }
  data.teams.forEach((_, i) => {el(`bridge-hits-${i}`).onchange = () => {const next = assumed.slice(); next[i] = Number(el(`bridge-hits-${i}`).value); setHits(next);};});
  el('bridge-calculate').onclick = () => setStage(1); el('bridge-compare').onclick = () => setStage(2); el('bridge-compare-all').onclick = () => setStage(3);
  el('bridge-reset').onclick = () => setHits(Array(6).fill(0));
  globalThis.MIA_FIXED_MEAN = Object.freeze({model, render, setHits, setStage, prediction, historyPrediction, candidates: () => candidates.map(row => ({...row, hits: row.hits.slice()}))});
})();
