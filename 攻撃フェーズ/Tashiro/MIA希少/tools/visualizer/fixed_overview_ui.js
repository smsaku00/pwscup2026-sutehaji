/* Read six-Team responses by row, then follow one Team's five-entry column. */
(function () {
  'use strict';
  const el = id => document.getElementById(id);
  const lesson = globalThis.MIA_FIXED_LESSON;
  let memberships = data.demo_masks.slice(), selectedTeam = 0, selectedRound = 0, view = 'overview', lastDepth = 0;
  const activeTeams = {1: 0, 2: 1};
  const label = mask => shared.maskIDs(mask, ids).join('＋');
  function assign(team, member) {
    if (!five.candidates(data.caps[team]).includes(member)) throw Error('チームの所属人数に合わない正解です。');
    const next = memberships.slice(), deferred = [];
    let used = member; next[team] = member;
    for (let t = 0; t < 6; t++) if (t !== team) {
      if (next[t] & used) deferred.push(t);
      else used |= next[t];
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
    const responses = data.query_masks.map(query => memberships.map(member => pop(query & member)));
    const columns = data.teams.map((_, t) => responses.map(hits => hits[t]));
    const decodedMembers = columns.map((column, t) => {
      const candidates = five.candidates(data.caps[t]).filter(member => data.query_masks.every((query, r) => pop(member & query) === column[r]));
      if (candidates.length !== 1) throw Error('5回の縦列から所属を一意に照合できません。');
      return candidates[0];
    });
    return {responses, columns, decodedMembers, rowSequences: responses.map(row => row.join('')),
      meanFractions: responses.map(row => globalThis.MIA_FIXED_VIEWER.meanExact(row))};
  }
  function render() {
    const state = model();
    const selectedClass = t => t === selectedTeam ? 'column-selected' : '';
    let table = '<thead><tr><th class="round-heading" scope="col">提出 ↓</th>';
    table += data.teams.map((team, t) => `<th scope="col" class="${selectedClass(t)}"><button data-overview-team="${t}" aria-pressed="${t === selectedTeam}" title="Team ${team}の5回を縦に読む">Team<br>${team}</button><span class="team-cap">所属${data.caps[t]}人</span></th>`).join('');
    table += '<th class="sequence-heading" scope="col">横の6桁 →</th></tr></thead><tbody>';
    table += state.responses.map((row, r) => `<tr data-overview-row="${r}" class="${r === selectedRound ? 'row-selected' : ''}"><th scope="row"><button data-overview-round="${r}" aria-pressed="${r === selectedRound}" title="提出${r + 1}の6チームを横に読む">提出${r + 1} →</button></th>${row.map((hit, t) => `<td data-overview-cell-team="${t}" data-overview-cell-round="${r}" data-hit="${hit}" class="${selectedClass(t)}${t === selectedTeam && r === selectedRound ? ' intersection' : ''}" title="提出${r + 1}・Team ${data.teams[t]}：${hit}人">${hit}</td>`).join('')}<td class="sequence"><code>${state.rowSequences[r]}</code></td></tr>`).join('');
    table += '</tbody><tfoot><tr class="signature-row"><th scope="row">縦の5桁 ↓</th>' + state.columns.map((column, t) => `<td data-overview-column="${t}" class="${selectedClass(t)}">${column.join('')}</td>`).join('') + '<td>各チームの<br>5回分</td></tr>';
    table += '<tr><th scope="row">所属ID</th>' + state.decodedMembers.map((member, t) => `<td data-overview-member="${t}" class="${selectedClass(t)}">${label(member)}</td>`).join('') + '<td>特定</td></tr></tfoot>';
    el('overview-table').innerHTML = table;
    const r = selectedRound, t = selectedTeam, sequence = state.columns[t].join('');
    el('row-reader').innerHTML = `<b>横 → 提出${r + 1}：</b><code>${state.rowSequences[r]}</code>（6チーム分）<span class="mean-detail">希少平均点（説明用）${state.meanFractions[r]} → 6チームの人数</span>`;
    el('column-reader').innerHTML = `<b>縦 ↓ Team ${data.teams[t]}：</b><code>${state.columns[t].join(' ↓ ')}</code><br><span class="vertical-arrow">5回分 <code>${sequence}</code> → ID ${label(state.decodedMembers[t])} → 右の5枚</span>`;
    el('selected-team-caption').textContent = `② Team ${data.teams[t]}の縦5桁 ${sequence} を重ねる`;
    el('lesson-workspace').setAttribute('data-view', view);
    el('show-overview').setAttribute('aria-pressed', String(view === 'overview'));
    el('show-cards').setAttribute('aria-pressed', String(view === 'cards'));
    for (let i = 0; i < 5; i++) el(`glass-card-${i + 1}`).classList.toggle('row-focus', i === r);
  }
  function fromLesson({cap, member, depth}) {
    selectedTeam = activeTeams[cap];
    const changed = memberships[selectedTeam] !== member;
    if (changed) memberships = assign(selectedTeam, member);
    if (changed || depth !== lastDepth) selectedRound = depth ? depth - 1 : 0;
    lastDepth = depth;
    render();
  }
  function selectTeam(team) {
    if (!Number.isInteger(team) || team < 0 || team >= 6) throw Error('対象のチームを選んでください。');
    const r = selectedRound;
    activeTeams[data.caps[team]] = team; selectedTeam = team;
    lesson.setAnswer(data.caps[team], memberships[team]);
    selectedRound = r; render();
  }
  function selectRound(round) {
    if (!Number.isInteger(round) || round < 0 || round >= 5) throw Error('提出1〜5を選んでください。');
    selectedRound = round; render();
  }
  function setView(value) {
    if (!['overview', 'cards'].includes(value)) throw Error('表示を選んでください。');
    view = value; render(); lesson.place(true);
  }
  function reset() {
    memberships = data.demo_masks.slice(); activeTeams[1] = 0; activeTeams[2] = 1; selectedRound = 0; view = 'overview';
    lesson.setAnswer(2, memberships[1]); lesson.setAnswer(1, memberships[0]); render(); lesson.place(true);
  }
  el('overview-table').onclick = event => {
    const target = event.target.closest('[data-overview-team],[data-overview-round],[data-overview-cell-team]');
    if (!target) return;
    if (target.dataset.overviewTeam !== undefined) selectTeam(Number(target.dataset.overviewTeam));
    else if (target.dataset.overviewCellTeam !== undefined) {selectTeam(Number(target.dataset.overviewCellTeam)); selectRound(Number(target.dataset.overviewCellRound));}
    else selectRound(Number(target.dataset.overviewRound));
  };
  el('show-overview').onclick = () => setView('overview');
  el('show-cards').onclick = () => setView('cards');
  el('reset-overview').onclick = reset;
  globalThis.MIA_FIXED_OVERVIEW = Object.freeze({fromLesson, selectTeam, selectRound, setView, reset,
    snapshot: () => ({selectedTeam, selectedRound, view, memberships: memberships.slice(), depth: lastDepth, ...model()})});
  const current = lesson.values();
  fromLesson({cap: current.cap, member: current.masks[current.cap], depth: current.depth});
})();
