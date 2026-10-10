/* Reveal one six-Team row at a time; show only the observed prefix of each column. */
(function () {
  'use strict';
  const el = id => document.getElementById(id), lesson = globalThis.MIA_FIXED_LESSON;
  let memberships = data.demo_masks.slice(), selectedTeam = 1, selectedRound = 0, view = 'intro', depth = 0, revealedCount = 0;
  const activeTeams = {1: 0, 2: 1};
  const label = mask => shared.maskIDs(mask, ids).join('＋');
  function assign(team, member) {
    if (!five.candidates(data.caps[team]).includes(member)) throw Error('チームの所属人数に合うIDを選んでください。');
    const next = memberships.slice(), deferred = []; let used = member; next[team] = member;
    for (let t = 0; t < 6; t++) if (t !== team) {
      if (next[t] & used) deferred.push(t); else used |= next[t];
    }
    for (const t of deferred) {
      let chosen = 0, needed = data.caps[t];
      for (let i = 0; i < ids.length && needed; i++) if (!(used >> i & 1)) {chosen |= 1 << i; needed--;}
      if (needed) throw Error('重複しない説明例を作れません。');
      next[t] = chosen; used |= chosen;
    }
    return next;
  }
  function model() {
    const responses = data.query_masks.map((query, r) => r < revealedCount ? memberships.map(member => pop(query & member)) : null);
    const columns = data.teams.map((_, t) => responses.map(row => row === null ? null : row[t]));
    const candidateSets = data.caps.map((cap, t) => five.candidates(cap).filter(member => data.query_masks.slice(0, depth).every((query, r) => pop(member & query) === pop(memberships[t] & query))));
    return {responses, columns, candidateSets, counts: candidateSets.map(set => set.length),
      rowSequences: responses.map(row => row === null ? '······' : row.join('')),
      meanFractions: responses.map(row => row === null ? null : globalThis.MIA_FIXED_VIEWER.meanExact(row))};
  }
  function render() {
    const state = model(), selected = t => t === selectedTeam ? 'column-selected' : '';
    let table = '<thead><tr><th class="round-heading" scope="col">提出 ↓</th>';
    table += data.teams.map((team, t) => `<th scope="col" class="${selected(t)}"><button data-overview-team="${t}" aria-pressed="${t === selectedTeam}" title="Team ${team}の候補を絞る">Team<br>${team}</button><span class="team-cap">所属${data.caps[t]}人</span></th>`).join('');
    table += '<th class="sequence-heading" scope="col">横の6桁 →</th></tr></thead><tbody>';
    table += state.responses.map((row, r) => `<tr data-overview-row="${r}" class="${r === selectedRound ? 'row-selected' : ''}"><th scope="row"><button data-overview-round="${r}" aria-pressed="${r === selectedRound}">提出${r + 1} →</button></th>${data.teams.map((_, t) => `<td data-overview-cell-team="${t}" data-overview-cell-round="${r}" data-pair-selectable="1" data-hit="${row === null ? '' : row[t]}" class="${selected(t)}${row === null ? ' unknown' : ''}${t === selectedTeam && r === selectedRound ? ' intersection' : ''}">${row === null ? '·' : row[t]}</td>`).join('')}<td class="sequence"><code>${state.rowSequences[r]}</code></td></tr>`).join('');
    table += '</tbody><tfoot><tr class="signature-row"><th scope="row">縦の結果 ↓</th>' + state.columns.map((column, t) => `<td data-overview-column="${t}" class="${selected(t)}">${column.map(n => n === null ? '·' : n).join('')}</td>`).join('') + '<td>表示済みの<br>結果だけ</td></tr>';
    table += '<tr><th scope="row">絞込済みの<br>候補</th>' + state.candidateSets.map((set, t) => `<td data-overview-count="${t}" class="${selected(t)}">${set.length === 1 ? 'ID ' + label(set[0]) : set.length + (data.caps[t] === 2 ? '組' : '候補')}</td>`).join('') + '<td>提出ごとに<br>減る</td></tr></tfoot>';
    el('overview-table').innerHTML = table;
    const r = selectedRound, t = selectedTeam;
    el('row-reader').innerHTML = state.responses[r] === null ? `<b>提出${r + 1}：</b>結果はまだ表示していません。`
      : `<b>横 → 提出${r + 1}：</b><code>${state.rowSequences[r]}</code>（6チーム分）<span class="mean-detail">説明用の希少平均点 ${state.meanFractions[r]} → 6チームの人数</span>`;
    el('column-reader').innerHTML = `<b>縦 ↓ Team ${data.teams[t]}：</b><code>${state.columns[t].map(n => n === null ? '·' : n).join(' ↓ ')}</code><br><span class="vertical-arrow">表示された結果で、②の${data.caps[t] === 2 ? '2人組の線' : '1人の候補行'}を絞ります。</span>`;
    el('lesson-team').innerHTML = data.teams.map((team, i) => `<option value="${team}"${i === t ? ' selected' : ''}>Team ${team}（所属${data.caps[i]}人）</option>`).join('');
    el('lesson-team').value = String(data.teams[t]);
    el('selected-team-caption').textContent = `② Team ${data.teams[t]}：${data.caps[t] === 2 ? '2人組を線で絞る' : '1人の候補行を絞る'}`;
    el('lesson-workspace').setAttribute('data-view', view);
    el('show-intro').setAttribute('aria-pressed', String(view === 'intro'));
    el('show-assignment').setAttribute('aria-pressed', String(view === 'assignment'));
    el('show-overview').setAttribute('aria-pressed', String(view === 'overview'));
    el('show-cards').setAttribute('aria-pressed', String(view === 'cards'));
    el('show-both').setAttribute('aria-pressed', String(view === 'both'));
    el('overview-panel').hidden = !['overview', 'both'].includes(view);
    el('pairs-panel').hidden = !['cards', 'both'].includes(view);
    el('assignment-panel').hidden = view !== 'assignment';
    el('intro-panel').hidden = view !== 'intro';
    el('lesson-controls').hidden = ['assignment', 'intro'].includes(view);
    el('lesson-actions').hidden = ['assignment', 'intro'].includes(view);
    for (let i = 0; i < 5; i++) el(`record-heading-${i + 1}`).classList.toggle('row-focus', i === r);
  }
  function fromLesson(state) {
    if (data.caps[selectedTeam] !== state.cap) selectedTeam = activeTeams[state.cap];
    const changed = memberships[selectedTeam] !== state.member;
    if (changed) memberships = assign(selectedTeam, state.member);
    if (changed || depth !== state.depth || revealedCount !== state.revealedCount) selectedRound = state.revealedCount ? state.revealedCount - 1 : 0;
    depth = state.depth; revealedCount = state.revealedCount; render();
  }
  function selectTeam(team) {
    if (!Number.isInteger(team) || team < 0 || team >= 6) throw Error('Teamを選んでください。');
    selectedTeam = team; activeTeams[data.caps[team]] = team; lesson.setAnswer(data.caps[team], memberships[team]); render();
  }
  function selectCapacity(capacity) {
    if (![1, 2].includes(capacity)) throw Error('所属1人か2人を選んでください。');
    selectTeam(activeTeams[capacity]);
  }
  function selectRound(round) {
    if (!Number.isInteger(round) || round < 0 || round >= 5) throw Error('提出1〜5を選んでください。');
    selectedRound = round; render();
  }
  function setView(value) {
    if (!['intro', 'assignment', 'overview', 'cards', 'both'].includes(value)) throw Error('表示を選んでください。');
    if (['assignment', 'intro'].includes(value)) {lesson.pause(); lesson.render();}
    if (value !== 'assignment') globalThis.MIA_FIXED_ASSIGNMENT.stop();
    view = value; render();
    if (view === 'assignment') globalThis.MIA_FIXED_ASSIGNMENT.render();
  }
  function reset() {
    memberships = data.demo_masks.slice(); activeTeams[1] = 0; activeTeams[2] = 1; selectedTeam = 1; selectedRound = 0; view = 'overview'; lesson.setAnswer(2, lesson.defaultMember); render();
  }
  el('overview-table').onclick = event => {
    const target = event.target.closest('[data-overview-team],[data-overview-round],[data-overview-cell-team]'); if (!target) return;
    if (target.dataset.overviewTeam !== undefined) selectTeam(Number(target.dataset.overviewTeam));
    else if (target.dataset.overviewCellTeam !== undefined) {
      const t = Number(target.dataset.overviewCellTeam);
      selectTeam(t); selectRound(Number(target.dataset.overviewCellRound));
    } else selectRound(Number(target.dataset.overviewRound));
  };
  el('lesson-team').onchange = () => selectTeam(data.teams.indexOf(Number(el('lesson-team').value)));
  el('show-intro').onclick = () => setView('intro'); el('show-assignment').onclick = () => setView('assignment'); el('show-overview').onclick = () => setView('overview'); el('show-cards').onclick = () => setView('cards'); el('show-both').onclick = () => setView('both'); el('reset-overview').onclick = reset;
  globalThis.MIA_FIXED_OVERVIEW = Object.freeze({fromLesson, selectTeam, selectCapacity, selectRound, setView, reset,
    snapshot: () => ({selectedTeam, selectedRound, view, memberships: memberships.slice(), depth, revealedCount, ...model()})});
  const state = lesson.snapshot(); fromLesson({cap: state.cap, member: state.member, depth: state.depth, revealedCount: state.revealedCount});
})();
