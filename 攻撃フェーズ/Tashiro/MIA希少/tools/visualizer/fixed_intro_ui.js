/* The historical two submissions identify Team counts; the fixed code is a teaching example. */
(function () {
  'use strict';
  const el = id => document.getElementById(id), history = data.intro, conditions = history.conditions;
  let page = 'roc', stage = 0, caseID = history.cases[0].case_id, team = '6';
  let mode = 'population', members = 1, high = 11, hits = 1, plot = 'full';
  const decimal = n => Number(n.toPrecision(12)).toString();
  function rocModel() {
    const x = (high - hits) / (24 - members), y = hits / members;
    return {mode, members, high, hits, highMembers: hits, highNonmembers: high - hits,
      lowMembers: members - hits, lowNonmembers: 24 - members - high + hits,
      points: [[0, 0], [x, y], [1, 1]], targetFPR: .01, score: .01 * y / x, plot};
  }
  function teamScore(n, m) {return m ? (24 - m) / (100 * (n - m)) : null;}
  function mean(row, round) {
    const sizes = conditions[round === 1 ? 'round1_sizes' : 'round2_sizes'];
    const active = conditions.teams.map((t, i) => ({n: sizes[i], m: row.counts[t]})).filter(x => x.m);
    return active.reduce((sum, x) => sum + teamScore(x.n, x.m), 0) / active.length;
  }
  function historyModel() {
    const row = history.cases.find(x => x.case_id === caseID);
    return {stage, caseID, team, counts: {...row.counts}, activeTeams: row.active_teams, totalMembers: row.total_members,
      predicted: [mean(row, 1), mean(row, 2)], observed: [stage >= 1 ? conditions.round1_rare_score : null, stage >= 2 ? conditions.round2_rare_score : null],
      remaining: stage === 0 ? null : stage === 1 ? history.cases.length : 1,
      matchesSecond: Math.abs(mean(row, 2) - conditions.round2_rare_score) < conditions.score_match_tolerance};
  }
  function graph(model, zoom) {
    const left = 56, top = 30, width = 330, height = 190;
    const xmax = zoom ? .02 : 1, ymax = zoom ? Math.max(.06, Math.ceil(model.score * 200) / 100 + .01) : 1;
    const px = x => left + x / xmax * width, py = y => top + height - y / ymax * height;
    const points = zoom ? [[0, 0], [xmax, model.points[1][1] * xmax / model.points[1][0]]] : model.points;
    const path = points.map((p, i) => `${i ? 'L' : 'M'}${px(p[0])},${py(p[1])}`).join(' ');
    const grid = [0, .5, 1].map(f => `<path d="M${px(f * xmax)},${top} V${top + height} M${left},${py(f * ymax)} H${left + width}" class="roc-grid"/><text x="${px(f * xmax)}" y="${top + height + 19}" text-anchor="middle">${zoom ? (f * xmax * 100).toFixed(0) + '%' : f * 100 + '%'}</text><text x="${left - 8}" y="${py(f * ymax) + 4}" text-anchor="end">${(f * ymax * 100).toFixed(zoom ? 1 : 0)}%</text>`).join('');
    const endpoint = zoom ? '' : `<circle cx="${px(model.points[1][0])}" cy="${py(model.points[1][1])}" r="5" class="roc-endpoint"/><text x="${Math.min(370, Math.max(145, px(model.points[1][0])))}" y="${Math.min(205, py(model.points[1][1]) + 23)}" text-anchor="middle">高側を全員選ぶ点</text>`;
    return `<svg viewBox="0 0 420 280" role="img" aria-label="${zoom ? 'FPRの1%付近を拡大' : 'ROCの全体'}、高側の点はFPR ${decimal(model.points[1][0])}、TPR ${decimal(model.points[1][1])}。FPR 0.01のTPRは${decimal(model.score)}"><text x="56" y="17" class="roc-axis-title">TPR：所属者を拾う割合 ↑</text>${grid}<path d="M${left},${top} V${top + height} H${left + width}" class="roc-axis"/><path d="${path}" class="roc-curve"/>${endpoint}<path d="M${px(.01)},${top} V${top + height}" class="roc-target"/><path d="M${left},${py(model.score)} H${px(.01)}" class="roc-reading"/><circle data-roc-score="${zoom ? 'zoom' : 'full'}" data-fpr="0.01" data-tpr="${model.score}" cx="${px(.01)}" cy="${py(model.score)}" r="5" class="roc-score"/><text x="${zoom ? px(.01) + 10 : 82}" y="${zoom ? Math.max(43, py(model.score) - 12) : 252}" class="roc-score-label">FPR 1% → TPR ${(model.score * 100).toFixed(3)}%</text><text x="386" y="272" text-anchor="end">FPR：非所属者まで選ぶ割合 →</text></svg>`;
  }
  function people(positive, negative) {
    return Array.from({length: positive}, () => '<span class="roc-person member" aria-label="所属者">●</span>').join('') + Array.from({length: negative}, () => '<span class="roc-person nonmember" aria-label="非所属者">○</span>').join('');
  }
  function renderROC() {
    const state = rocModel();
    el('roc-mode').value = mode; el('roc-members').value = String(members);
    el('roc-high').min = mode === 'population' ? '11' : '6'; el('roc-high').max = mode === 'population' ? '24' : '12'; el('roc-high').value = String(high);
    el('roc-high-value').textContent = `${high}人`;
    el('roc-hits').innerHTML = Array.from({length: members + 1}, (_, i) => `<option value="${i}"${i === hits ? ' selected' : ''}>${i}人</option>`).join('');
    el('roc-hits').value = String(hits); el('roc-hits').disabled = mode === 'population';
    el('roc-high-people').innerHTML = people(state.highMembers, state.highNonmembers);
    el('roc-low-people').innerHTML = people(state.lowMembers, state.lowNonmembers);
    el('roc-high-caption').textContent = `高側 0.4：所属${hits}人 ＋ 非所属${high - hits}人`;
    el('roc-low-caption').textContent = `低側 0.1：所属${members - hits}人 ＋ 非所属${state.lowNonmembers}人`;
    el('roc-full').innerHTML = graph(state, false); el('roc-zoom').innerHTML = graph(state, true);
    el('roc-plots').setAttribute('data-plot', plot);
    el('roc-show-full').setAttribute('aria-pressed', String(plot === 'full')); el('roc-show-zoom').setAttribute('aria-pressed', String(plot === 'zoom'));
    el('roc-formula').innerHTML = `<span><b>高側を全員選ぶ点</b><br>FPR＝${high - hits}/${24 - members}　TPR＝${hits}/${members}</span><span><b>1%まで直線で読む</b><br>0.01 × (${hits}/${members}) ÷ (${high - hits}/${24 - members})<br><strong>TPR＝${decimal(state.score)}</strong></span>`;
    el('roc-explanation').textContent = mode === 'population'
      ? `第1・2回は所属候補11人を全員高側へ。所属者も全員高側なのでTPRの終点は1。高側の非所属者を混ぜる人数で傾きが変わり、人数1人・2人の予測点数を計算できます。`
      : `次はm＝${members}人が分かったチームを考えます。高側の所属者hを0〜${members}人と仮置きすると予測TPRが変わります。返却される平均点と照合して、各チームのhを読み分けます。`;
  }
  function casePlot(selected) {
    const ordered = history.cases.slice().sort((a, b) => a.round2_score - b.round2_score), values = ordered.map(x => x.round2_score);
    const lo = Math.min(...values) - .0002, hi = Math.max(...values) + .0002;
    const px = value => 42 + (value - lo) / (hi - lo) * 550;
    const circles = ordered.map((row, i) => `<circle data-history-case="${row.case_id}" data-predicted-score="${row.round2_score}" class="history-dot${stage === 2 && row.case_id !== history.winner ? ' eliminated' : ''}${row.case_id === selected.case_id ? ' selected' : ''}" cx="${px(row.round2_score)}" cy="${30 + i % 7 * 15}" r="4"><title>${row.case_id}：${row.round2_score_exact}</title></circle>`).join('');
    const marker = stage === 2 ? `<path d="M${px(conditions.round2_rare_score)},16 V134" class="history-observed"/><text x="${Math.max(150, Math.min(500, px(conditions.round2_rare_score)))}" y="158" text-anchor="middle">第2回の実値に一致：${history.winner}</text>` : '<text x="320" y="158" text-anchor="middle">どの点が返るかで、102通りを区別する</text>';
    return `<svg viewBox="0 0 640 185" role="img" aria-label="102通りの第2回予測点数。点数はすべて異なる。${stage === 2 ? '実値と一致するのはN015のみ。' : ''}"><path d="M42,134 H592" class="roc-axis"/>${circles}${marker}<text x="42" y="177">${decimal(lo)}</text><text x="592" y="177" text-anchor="end">${decimal(hi)} → 第2回予測点</text></svg>`;
  }
  function renderHistory() {
    const state = historyModel(), row = history.cases.find(x => x.case_id === caseID), i = conditions.teams.indexOf(team);
    const n1 = conditions.round1_sizes[i], n2 = conditions.round2_sizes[i];
    el('history-team').innerHTML = conditions.teams.map(t => `<option value="${t}"${t === team ? ' selected' : ''}>Team ${t}</option>`).join(''); el('history-team').value = team;
    el('history-case').innerHTML = history.cases.map(x => `<option value="${x.case_id}"${x.case_id === caseID ? ' selected' : ''}>${x.case_id}（${x.active_teams}チーム・${x.total_members}人）</option>`).join(''); el('history-case').value = caseID;
    el('history-case-controls').hidden = stage === 0;
    for (let s = 0; s < 3; s++) el(`history-stage-${s}`).setAttribute('aria-pressed', String(s === stage));
    el('history-setting').innerHTML = `<h3>Team ${team} の提出設定</h3>` + [n1, n2].map((n, r) => `<div class="history-submission"><b>第${r + 1}回</b><span class="history-unknown">候補11人<br>全員0.4</span><b>＋</b><span>既知非所属${n - 11}人<br>同じ0.4</span><b>＝ 高側${n}人</b><small>残る${24 - n}人は0.1</small></div>`).join('');
    el('history-hypotheses').innerHTML = '<table><caption>所属人数を仮置きした、そのチームの予測TPR</caption><thead><tr><th>仮定</th><th>第1回</th><th>第2回</th></tr></thead><tbody><tr><td>0人</td><td colspan="2">平均の対象から外れる</td></tr>' + [1, 2].map(m => `<tr><td>${m}人</td><td>${decimal(teamScore(n1, m))}</td><td>${decimal(teamScore(n2, m))}</td></tr>`).join('') + '</tbody></table>';
    const headings = ['まず、22チームの人数を仮置きする', '第1回の平均点に合う人数配置は102通り', '第2回も合う配置は1通り：N015'];
    el('history-heading').textContent = headings[stage];
    el('history-score').innerHTML = stage === 0 ? '<b>条件：各チーム0・1・2人、合計11人以下</b><span>該当チーム数を6つと決めずに探索。個人IDは全員同点なので区別できません。</span>' : `<b>第1回の実値 ${conditions.round1_rare_score}</b>` + (stage === 2 ? `<b>第2回の実値 ${conditions.round2_rare_score}</b>` : '<span>この点数だけでは、6チームの配置も7チームの配置も残ります。</span>');
    el('history-team-grid').innerHTML = conditions.teams.map(t => `<div class="history-team-cell${stage && row.counts[t] ? ' active' : ''}${stage === 2 && !state.matchesSecond ? ' comparison' : ''}"><span>Team ${t}</span><b>${stage ? row.counts[t] + '人' : '0 / 1 / 2 ?'}</b></div>`).join('');
    el('history-selected-caption').textContent = stage === 0 ? '人数はまだ不明。全22チームを対象に考えます。' : stage === 1 ? `${caseID}を仮置き：${row.active_teams}チーム・${row.total_members}人（102通りのうちの1例）` : `${caseID}：${state.matchesSecond ? '第2回まで一致 → 残る人数配置' : '第2回が不一致 → 残らない比較例'}`;
    el('history-case-plot').hidden = stage === 0;
    el('history-case-plot').innerHTML = casePlot(row);
    el('history-predicted').hidden = stage === 0;
    el('history-predicted').innerHTML = `<span>選んだ配置の予測平均：第1回 <b>${decimal(state.predicted[0])}</b> ／ 第2回 <b>${row.round2_score_exact}</b> ≈ ${decimal(state.predicted[1])}</span>`;
    el('history-conclusion').textContent = stage === 0 ? '実際に返るのは、所属が1人以上いるチームだけの平均点です。各チームの点数が個別に返る前提ではありません。' : stage === 1 ? '全102配置で第1回の平均点は同じ。第2回は、既知非所属者を混ぜる人数を変え、102通りの予測平均点をすべて異なる値にしました。' : '第2回終了時点：Team 6・11・17は各1人、10・16・25は各2人、他16チームは0人。合計9人。誰がどこに所属するかは、この2回だけでは未特定です。';
  }
  function bridgeModel() {
    const state = globalThis.MIA_FIXED_LESSON.snapshot();
    const teamIndex = globalThis.MIA_FIXED_OVERVIEW?.snapshot().selectedTeam ?? 1;
    const caps = data.caps, sample = data.high.map((n, i) => Array.from({length: caps[i] + 1}, (_, h) => .01 * h / caps[i] * (24 - caps[i]) / (n - h)));
    return {teamIndex, capacity: state.cap, individualScores: sample, responses: [0, 1, 0, 1, 1, 1], mean: 73 / 6750};
  }
  function renderBridge() {
    const state = bridgeModel();
    el('bridge-table').innerHTML = '<thead><tr><th>人数が分かったチーム</th><th>所属人数 m</th><th>高側合計 n</th><th>高側の所属者 h → 予測TPR</th></tr></thead><tbody>' + data.teams.map((t, i) => `<tr><td>Team ${t}</td><td>${data.caps[i]}人</td><td>${data.high[i]}人</td><td>${state.individualScores[i].map((score, h) => `${h}人 → ${decimal(score)}`).join(' ／ ')}</td></tr>`).join('') + '</tbody>';
    el('bridge-example').innerHTML = `<span class="bridge-mean">教材の平均点<br><b>73/6750</b><small> ≈ ${decimal(state.mean)}</small></span><span class="bridge-arrow">→</span><span><b>6チームの高側人数</b><code>0　1　0　1　1　1</code><small>Team 6・10・11・16・17・25 の順</small></span>`;
  }
  function render() {
    el('intro-page-roc').hidden = page !== 'roc'; el('intro-page-history').hidden = page !== 'history'; el('intro-page-bridge').hidden = page !== 'bridge';
    for (const value of ['roc', 'history', 'bridge']) el(`intro-tab-${value}`).setAttribute('aria-pressed', String(page === value));
    el('intro-prev').disabled = page === 'roc';
    el('intro-next').textContent = page === 'roc' ? '第1・2回の流れへ →' : page === 'bridge' ? '⓪ 5回に振り分けるへ →' : ['第1回の結果を見る →', '第2回で絞る →', '個人の特定へつなげる →'][stage];
    renderROC(); renderHistory(); renderBridge();
  }
  function setPage(value) {if (!['roc', 'history', 'bridge'].includes(value)) throw Error('説明ページを選んでください。'); page = value; render();}
  function setStage(value) {if (![0, 1, 2].includes(value)) throw Error('第1・2回の段階を選んでください。'); stage = value; if (stage === 2) caseID = history.winner; render();}
  function setCase(value) {if (!history.cases.some(x => x.case_id === value)) throw Error('人数配置を選んでください。'); caseID = value; render();}
  function setROC(next) {
    const values = {...rocModel(), ...next};
    if (!['population', 'subset'].includes(values.mode) || ![1, 2].includes(values.members) || !Number.isInteger(values.high) || values.high < (values.mode === 'population' ? 11 : 6) || values.high > (values.mode === 'population' ? 24 : 12) || !Number.isInteger(values.hits) || values.hits < 0 || values.hits > values.members) throw Error('ROCの人数設定を確認してください。');
    mode = values.mode; members = values.members; high = values.high; hits = mode === 'population' ? members : values.hits; render();
  }
  el('roc-mode').onchange = () => setROC(el('roc-mode').value === 'population' ? {mode: 'population', members: 1, high: 11, hits: 1} : {mode: 'subset', members: 2, high: 7, hits: 1});
  el('roc-members').onchange = () => {const m = Number(el('roc-members').value); setROC({members: m, hits: Math.min(m, hits)});};
  el('roc-high').oninput = () => setROC({high: Number(el('roc-high').value)});
  el('roc-hits').onchange = () => setROC({hits: Number(el('roc-hits').value)});
  el('roc-show-full').onclick = () => {plot = 'full'; render();}; el('roc-show-zoom').onclick = () => {plot = 'zoom'; render();};
  for (const value of ['roc', 'history', 'bridge']) el(`intro-tab-${value}`).onclick = () => setPage(value);
  for (let s = 0; s < 3; s++) el(`history-stage-${s}`).onclick = () => setStage(s);
  el('history-team').onchange = () => {team = el('history-team').value; render();}; el('history-case').onchange = () => setCase(el('history-case').value);
  el('history-case-plot').onclick = event => {const target = event.target.closest('[data-history-case]'); if (target) setCase(target.dataset.historyCase);};
  el('intro-next').onclick = () => {if (page === 'roc') setPage('history'); else if (page === 'history' && stage < 2) setStage(stage + 1); else if (page === 'history') setPage('bridge'); else globalThis.MIA_FIXED_OVERVIEW.setView('assignment');};
  el('intro-prev').onclick = () => {if (page === 'history' && stage > 0) setStage(stage - 1); else setPage(page === 'bridge' ? 'history' : 'roc');};
  globalThis.MIA_FIXED_INTRO = Object.freeze({setPage, setStage, setCase, setROC, render, rocModel, historyModel, bridgeModel, snapshot: () => ({page, stage, caseID, team, roc: rocModel(), historical: historyModel()})});
  render();
})();
