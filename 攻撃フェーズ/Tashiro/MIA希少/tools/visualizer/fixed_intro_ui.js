/* The historical two submissions identify Team counts; the fixed code is a teaching example. */
(function () {
  'use strict';
  const el = id => document.getElementById(id), history = data.intro, conditions = history.conditions;
  let page = 'roc', stage = 0, caseID = history.cases[0].case_id, team = '6';
  let historyView = 'mean', historyRound = 1;
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
      matchesSecond: Math.abs(mean(row, 2) - conditions.round2_rare_score) < conditions.score_match_tolerance,
      meanView: historyView, meanRound: historyRound, meanDetails: globalThis.MIA_FIXED_MEAN.historyPrediction(row, historyRound)};
  }
  function graph(model, zoom) {
    const left = 56, top = 60, width = 330, height = 160;
    const xmax = zoom ? .02 : 1, ymax = zoom ? Math.max(.06, Math.ceil(model.score * 200) / 100 + .01) : 1;
    const px = x => left + x / xmax * width, py = y => top + height - y / ymax * height;
    const points = zoom ? [[0, 0], [xmax, model.points[1][1] * xmax / model.points[1][0]]] : model.points;
    const path = points.map((p, i) => `${i ? 'L' : 'M'}${px(p[0])},${py(p[1])}`).join(' ');
    const grid = [0, .5, 1].map(f => `<path d="M${px(f * xmax)},${top} V${top + height} M${left},${py(f * ymax)} H${left + width}" class="roc-grid"/><text x="${px(f * xmax)}" y="${top + height + 19}" text-anchor="middle">${zoom ? (f * xmax * 100).toFixed(0) + '%' : f * 100 + '%'}</text><text x="${left - 8}" y="${py(f * ymax) + 4}" text-anchor="end">${(f * ymax * 100).toFixed(zoom ? 1 : 0)}%</text>`).join('');
    const badgeX = Math.min(186, Math.max(66, px(model.points[1][0]) - 100));
    const badgeY = model.points[1][1] > .5 ? py(model.points[1][1]) + 17 : py(model.points[1][1]) - 83;
    const endpoint = zoom ? '' : `<path d="M${px(model.points[1][0])},${py(model.points[1][1])} L${badgeX + 100},${badgeY + (model.points[1][1] > .5 ? 0 : 68)}" class="roc-count-leader"/><circle cx="${px(model.points[1][0])}" cy="${py(model.points[1][1])}" r="5" class="roc-endpoint"/><g class="roc-count-badge" data-roc-coordinate="high" data-x-numerator="${model.highNonmembers}" data-x-denominator="${24 - model.members}" data-y-numerator="${model.highMembers}" data-y-denominator="${model.members}"><rect x="${badgeX}" y="${badgeY}" width="200" height="68" rx="7"/><text x="${badgeX + 100}" y="${badgeY + 17}" text-anchor="middle">高側を全員選ぶ点：実人数</text><text x="${badgeX + 100}" y="${badgeY + 38}" text-anchor="middle" class="roc-count-value">x＝${model.highNonmembers}人 / ${24 - model.members}人</text><text x="${badgeX + 100}" y="${badgeY + 58}" text-anchor="middle" class="roc-count-value">y＝${model.highMembers}人 / ${model.members}人</text></g>`;
    const scoreLabel = zoom
      ? `<text x="221" y="252" text-anchor="middle" class="roc-score-label">x＝1/100＝1% → y＝${(model.score * 100).toFixed(3)}%</text><text x="221" y="269" text-anchor="middle">1%の点は、実人数の点を結ぶ直線で補間</text>`
      : `<text x="221" y="252" text-anchor="middle" class="roc-score-label">1%の点（補間）→ TPR ${(model.score * 100).toFixed(3)}%</text>`;
    return `<svg viewBox="0 0 420 320" role="img" aria-label="${zoom ? 'FPRの1%付近を拡大' : 'ROCの全体'}。xは選んだ非所属者数を全非所属者${24 - model.members}人で割る。yは拾った所属者数を全所属者${model.members}人で割る。高側の点のxは${model.highNonmembers}人/${24 - model.members}人、yは${model.highMembers}人/${model.members}人。FPR 0.01の補間TPRは${decimal(model.score)}"><text x="56" y="17" class="roc-axis-title">y（TPR）＝ 拾った所属者の人数</text><text x="56" y="35" class="roc-axis-title">　　　　　/ 全所属者の人数（${model.members}人） ↑</text>${grid}<path d="M${left},${top} V${top + height} H${left + width}" class="roc-axis"/><path d="${path}" class="roc-curve"/><path d="M${px(.01)},${top} V${top + height}" class="roc-target"/><path d="M${left},${py(model.score)} H${px(.01)}" class="roc-reading"/>${endpoint}<circle data-roc-score="${zoom ? 'zoom' : 'full'}" data-fpr="0.01" data-tpr="${model.score}" cx="${px(.01)}" cy="${py(model.score)}" r="5" class="roc-score"/>${scoreLabel}<text x="56" y="291" class="roc-axis-title">x（FPR）＝ 選んだ非所属者の人数</text><text x="56" y="309" class="roc-axis-title">　　　　　/ 全非所属者の人数（${24 - model.members}人） →</text></svg>`;
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
    el('history-team-grid').hidden = stage > 0 && historyView === 'mean';
    el('history-team-grid').innerHTML = conditions.teams.map(t => `<div class="history-team-cell${stage && row.counts[t] ? ' active' : ''}${stage === 2 && !state.matchesSecond ? ' comparison' : ''}"><span>Team ${t}</span><b>${stage ? row.counts[t] + '人' : '0 / 1 / 2 ?'}</b></div>`).join('');
    el('history-selected-caption').textContent = stage === 0 ? '人数はまだ不明。全22チームを対象に考えます。' : stage === 1 ? `${caseID}を仮置き：${row.active_teams}チーム・${row.total_members}人（102通りのうちの1例）` : `${caseID}：${state.matchesSecond ? '第2回まで一致 → 残る人数配置' : '第2回が不一致 → 残らない比較例'}`;
    el('history-case-plot').hidden = stage === 0 || historyView === 'mean';
    el('history-case-plot').innerHTML = casePlot(row);
    el('history-predicted').hidden = stage === 0 || historyView === 'mean';
    el('history-predicted').innerHTML = `<span>選んだ配置の予測平均：第1回 <b>${decimal(state.predicted[0])}</b> ／ 第2回 <b>${row.round2_score_exact}</b> ≈ ${decimal(state.predicted[1])}</span>`;
    el('history-view-controls').hidden = stage === 0;
    el('history-show-mean').setAttribute('aria-pressed', String(historyView === 'mean')); el('history-show-configurations').setAttribute('aria-pressed', String(historyView === 'configurations'));
    el('history-mean-panel').hidden = stage === 0 || historyView !== 'mean';
    for (const r of [1, 2]) el(`history-mean-round-${r}`).setAttribute('aria-pressed', String(historyRound === r));
    const details = state.meanDetails, observed = state.observed[historyRound - 1];
    el('history-mean-contributions').innerHTML = details.contributions.map(x => `<div data-mean-team="${x.team}" data-mean-tpr="${x.exact}"><b>Team ${x.team}</b><span>仮定 m＝${x.members}人 ／ 高側 n＝${x.high}人</span><b>TPR＝${x.exact}</b><span> ≈ ${decimal(x.value)}</span></div>`).join('');
    const checks = state.predicted.map((value, r) => state.observed[r] === null ? '未表示' : Math.abs(value - state.observed[r]) < conditions.score_match_tolerance ? '一致' : '不一致');
    el('history-mean-equation').innerHTML = `<b>第${historyRound}回：mの配置を照合（h＝m）</b><br><span>この仮定ではK＝${details.denominator}チームに所属者がいる。m＝0の${details.excluded}チームは平均から外れる。</span><br>（${details.contributions.map(x => x.exact).join(' ＋ ')}）<br>÷ <b>${details.denominator}チーム</b> ＝ <b>${details.meanExact}</b> ≈ ${decimal(details.meanValue)}<br>` + (observed === null ? `第${historyRound}回の返却点はまだ表示していません。` : `<b>${Math.abs(details.meanValue - observed) < conditions.score_match_tolerance ? '一致' : '不一致'}</b>：第${historyRound}回の返却点 ${decimal(observed)} と照合`) + `<br><span class="history-joint-check">同じm配置を確認：第1回 ${checks[0]} ／ 第2回 ${checks[1]}${stage === 2 ? checks.every(value => value === '一致') ? ' → 両方一致する配置を残す' : ' → 不一致の配置を外す' : ''}</span>`;
    el('history-conclusion').textContent = stage === 0 ? 'ここでは、22チームの所属人数mを探します。各チームの点数は個別に返らず、m＞0のチームの平均点が返ります。同じmなら個人を入れ替えても同点です。' : stage === 1 ? '第1回の平均点に合うmの配置は102通り。この中にはK＝6とK＝7の仮定が残ります。同じm配置に第2回の設定nを当てはめ、2つの平均点の両方に合う配置を探します。' : '第2回でmが確定：Team 6・11・17は各1人、10・16・25は各2人、他16チームは0人。合計9人。次はmと対象6チームを固定し、提出対象の所属人数hを読んで個人IDを絞ります。';
  }
  function bridgeModel() {
    const state = globalThis.MIA_FIXED_LESSON.snapshot();
    const teamIndex = globalThis.MIA_FIXED_OVERVIEW?.snapshot().selectedTeam ?? 1;
    const caps = data.caps, sample = data.high.map((n, i) => Array.from({length: caps[i] + 1}, (_, h) => .01 * h / caps[i] * (24 - caps[i]) / (n - h)));
    return {teamIndex, capacity: state.cap, individualScores: sample, responses: [0, 1, 0, 1, 1, 1], mean: 73 / 6750, comparison: globalThis.MIA_FIXED_MEAN.model()};
  }
  function renderBridge() {
    globalThis.MIA_FIXED_MEAN.render();
  }
  function render() {
    el('intro-page-roc').hidden = page !== 'roc'; el('intro-page-history').hidden = page !== 'history'; el('intro-page-bridge').hidden = page !== 'bridge';
    for (const value of ['roc', 'history', 'bridge']) el(`intro-tab-${value}`).setAttribute('aria-pressed', String(page === value));
    el('intro-prev').disabled = page === 'roc';
    el('intro-next').textContent = page === 'roc' ? '第1・2回でmを特定する →' : page === 'bridge' ? ['仮置きしたhの平均を計算 →', '今回の返却点と照合 →', '216通りすべてを照合 →', '⓪ 5回に振り分けるへ →'][globalThis.MIA_FIXED_MEAN.model().stage] : ['第1回の平均点で絞る →', '第2回の平均点も使う →', 'mを固定して、その後のhを読む →'][stage];
    renderROC(); renderHistory(); renderBridge();
  }
  function setPage(value) {if (!['roc', 'history', 'bridge'].includes(value)) throw Error('説明ページを選んでください。'); page = value; render();}
  function setStage(value) {if (![0, 1, 2].includes(value)) throw Error('第1・2回の段階を選んでください。'); stage = value; if (stage > 0) {historyRound = stage; historyView = 'mean';} if (stage === 2) caseID = history.winner; render();}
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
  el('history-show-mean').onclick = () => {historyView = 'mean'; render();}; el('history-show-configurations').onclick = () => {historyView = 'configurations'; render();};
  for (const r of [1, 2]) el(`history-mean-round-${r}`).onclick = () => {historyRound = r; render();};
  el('history-team').onchange = () => {team = el('history-team').value; render();}; el('history-case').onchange = () => setCase(el('history-case').value);
  el('history-case-plot').onclick = event => {const target = event.target.closest('[data-history-case]'); if (target) setCase(target.dataset.historyCase);};
  el('intro-next').onclick = () => {if (page === 'roc') setPage('history'); else if (page === 'history' && stage < 2) setStage(stage + 1); else if (page === 'history') setPage('bridge'); else if (globalThis.MIA_FIXED_MEAN.model().stage < 3) globalThis.MIA_FIXED_MEAN.setStage(globalThis.MIA_FIXED_MEAN.model().stage + 1); else globalThis.MIA_FIXED_OVERVIEW.setView('assignment');};
  el('intro-prev').onclick = () => {if (page === 'bridge' && globalThis.MIA_FIXED_MEAN.model().stage > 0) globalThis.MIA_FIXED_MEAN.setStage(globalThis.MIA_FIXED_MEAN.model().stage - 1); else if (page === 'history' && stage > 0) setStage(stage - 1); else setPage(page === 'bridge' ? 'history' : 'roc');};
  globalThis.MIA_FIXED_INTRO = Object.freeze({setPage, setStage, setCase, setROC, render, rocModel, historyModel, bridgeModel, snapshot: () => ({page, stage, caseID, team, roc: rocModel(), historical: historyModel()})});
  render();
})();
