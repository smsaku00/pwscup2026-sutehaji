/* Filter single rows or unordered-pair edges using only revealed results. */
(function () {
  'use strict';
  const el = id => document.getElementById(id), pairs = five.candidates(2);
  const label = mask => shared.maskIDs(mask, ids).join('＋');
  const defaultMember = pairs.find(candidate => pop(candidate & data.query_masks[0]) === 0 && remaining(2, candidate, 4).length > 1);
  const masks = {1: data.lesson_defaults[1], 2: defaultMember};
  let cap = 2, speed = 'slow';
  const speeds = Object.freeze({slow: {result: 1600, fade: 1700, gap: 1000}, normal: {result: 800, fade: 900, gap: 500}, fast: {result: 360, fade: 680, gap: 200}});
  const unit = () => cap === 2 ? '組' : '人';
  let member = defaultMember, pairOrder = shared.maskIDs(member, ids);
  let depth = 0, phase = null, playing = false, busy = false, epoch = 0, focusedPair = null;
  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function trace(capacity, target) {
    if (!five.candidates(capacity).includes(target)) throw Error('所属人数に合う異なるIDを選んでください。');
    let before = five.candidates(capacity);
    return data.query_masks.map((query, step) => {
      const hit = pop(query & target), grouped = five.groups(before, query);
      const after = before.filter(candidate => pop(candidate & query) === hit);
      const row = {step, query, hit, before: before.slice(), after: after.slice(),
        groups: Array.from({length: capacity + 1}, (_, h) => grouped.get(h) || [])};
      before = after; return row;
    });
  }
  function remaining(capacity, target, count) {
    if (!Number.isInteger(count) || count < 0 || count > 5) throw Error('提出数は0〜5です。');
    const rows = trace(capacity, target);
    return five.candidates(capacity).filter(candidate => rows.slice(0, count).every(row => pop(candidate & row.query) === row.hit));
  }
  function snapshot() {
    const rows = trace(cap, member), kept = remaining(cap, member, depth);
    const removing = phase && phase.stage === 'removing';
    const removed = removing ? kept.filter(candidate => !rows[depth].after.includes(candidate)) : [];
    const after = removing ? rows[depth].after : kept;
    const revealedCount = depth + (phase ? 1 : 0);
    return {cap, member, depth, playing, busy, speed, phase: phase ? {...phase} : null, revealedCount,
      results: rows.map((row, r) => r < revealedCount ? row.hit : null),
      remainingMasks: kept.slice(), displayedMasks: kept.slice(), removedMasks: removed,
      activeIDs: ids.filter((_, i) => kept.some(candidate => candidate >> i & 1)),
      nextActiveIDs: ids.filter((_, i) => after.some(candidate => candidate >> i & 1)),
      counts: [five.candidates(cap).length, ...rows.slice(0, depth).map(row => row.after.length)], focusedPair};
  }
  function geometry(width = 520, height = 330) {
    const w = Math.max(0, Number(width) || 0), h = Math.max(0, Number(height) || 0);
    const x = w * 318 / 520, nodes = ids.map((id, i) => ({id, x, y: h * (i + .5) / ids.length}));
    const edges = pairs.map(mask => {
      const positions = ids.map((_, i) => i).filter(i => mask >> i & 1), [a, b] = positions;
      const control = x + w * (28 + (b - a) * 18) / 520;
      return {mask, a, b, start: nodes[a], end: nodes[b], control,
        d: `M ${x} ${nodes[a].y} C ${control} ${nodes[a].y} ${control} ${nodes[b].y} ${x} ${nodes[b].y}`};
    });
    return {width: w, height: h, nodes, edges};
  }
  function pickers() {
    [1, 2].forEach((number, i) => {
      const select = el(`answer-record-${number}`);
      select.innerHTML = ids.map(id => `<option value="${id}"${id === pairOrder[i] ? ' selected' : ''}${cap === 2 && id === pairOrder[1 - i] ? ' disabled' : ''}>ID ${id}</option>`).join('');
      select.value = String(pairOrder[i] === undefined ? ids[1] : pairOrder[i]);
    });
    el('answer-record-2-wrap').hidden = cap !== 2;
  }
  function timing() {const selected = speeds[speed]; return {...selected, fade: reduced ? 200 : selected.fade};}
  function render() {
    const state = snapshot(), rows = trace(cap, member), stage = phase?.stage;
    const r = phase ? depth : Math.max(0, depth - 1), current = rows[r];
    const removed = new Set(state.removedMasks), alive = new Set(state.activeIDs), nextAlive = new Set(state.nextActiveIDs);
    const unique = state.remainingMasks.length === 1 && !phase;
    if (focusedPair !== null && !state.displayedMasks.includes(focusedPair)) focusedPair = null;
    pickers();
    el('tab-single').setAttribute('aria-selected', String(cap === 1));
    el('tab-pair').setAttribute('aria-selected', String(cap === 2));
    el('playback-speed').value = speed;
    el('pairs-panel').setAttribute('data-cap', String(cap));
    el('pair-stage').setAttribute('data-cap', String(cap));
    el('pair-stage').style.setProperty('--fade-ms', `${timing().fade}ms`);
    el('line-label').textContent = cap === 2 ? '線＝2人の組' : '';
    el('mode-legend').textContent = cap === 2 ? '線の2人分を足す：0＋0＝0、0＋1＝1、1＋1＝2' : '各行の0・1と、提出結果の0・1を比べます。';
    el('interaction-legend').textContent = cap === 2 ? '線を押すと、その組の合計を確認できます。' : '結果と合わない行は、薄く残します。';
    el('result-strip').innerHTML = state.results.map((hit, i) => `<div class="result-chip${hit === null ? ' pending' : ''}${i === r && state.revealedCount ? ' current' : ''}" data-round="${i}" data-result="${hit === null ? '' : hit}" aria-label="提出${i + 1}：${hit === null ? '未表示' : hit + '人'}"><span>提出${i + 1}</span><b>${hit === null ? '—' : hit}<small>${hit === null ? '' : '人'}</small></b></div>`).join('');
    el('pair-count').textContent = stage === 'removing' ? `${state.remainingMasks.length} → ${current.after.length}${unit()}` : `残る${state.remainingMasks.length}${unit()}`;
    const rule = cap === 2 ? `${current.hit === 0 ? '0と0を結ぶ線' : current.hit === 1 ? '0と1を結ぶ線' : '1と1を結ぶ線'}を残します。`
      : `提出${r + 1}が${current.hit}の行を残し、${1 - current.hit}の行を候補から外します。`;
    el('filter-rule').textContent = !state.revealedCount ? `次の提出結果を表示すると、合わない${cap === 2 ? '線' : '行'}が${cap === 2 ? '消えます' : '薄くなります'}。`
      : `提出${r + 1}の結果：${current.hit}人。${rule}${stage === 'result' ? (playing ? 'まず結果を表示しています。' : '「この結果で候補を絞る」を押して進みます。') : stage === 'removing' ? `合わない${cap === 2 ? '線' : '行'}が${cap === 2 ? '消えます' : '薄くなります'}。` : ''}`;
    el('record-deck').innerHTML = ids.map((id, i) => {
      const visible = alive.has(id), fading = stage === 'removing' && visible && !nextAlive.has(id);
      const bits = data.query_masks.map(query => Number(Boolean(query >> i & 1)));
      return `<div class="pair-record${!visible ? ' excluded' : fading ? ' fading' : ''}${unique && visible ? ' unique' : ''}" data-record-id="${id}" data-active="${visible ? 1 : 0}" data-fading="${fading ? 1 : 0}" role="group" aria-label="ID ${id}：${!visible ? '候補外' : fading ? '候補外になります' : '候補'}"><b>ID ${id}</b>${bits.map((n, j) => `<span class="bit-${n ? 'one' : 'zero'}${state.revealedCount && j === r ? ' current-bit' : ''}" data-round="${j}" data-value="${n}" title="提出${j + 1}の対象${n ? '内' : '外'}">${n}</span>`).join('')}</div>`;
    }).join('');
    const layout = geometry(), edgeMap = new Map(layout.edges.map(edge => [edge.mask, edge]));
    const drawingOrder = (cap === 2 ? state.displayedMasks : []).filter(mask => mask !== focusedPair);
    if (focusedPair !== null) drawingOrder.push(focusedPair);
    el('pair-lines').innerHTML = drawingOrder.map(mask => {
      const edge = edgeMap.get(mask), fading = removed.has(mask), sum = state.revealedCount ? pop(mask & current.query) : null;
      return `<path class="pair-line${fading ? ' fading' : ''}${mask === focusedPair ? ' focused' : ''}${unique ? ' unique' : ''}" data-pair-mask="${mask}" data-fading="${fading ? 1 : 0}" data-current-sum="${sum === null ? '' : sum}" d="${edge.d}" tabindex="${fading ? -1 : 0}" role="button" aria-label="ID ${label(mask)}の組${sum === null ? '' : '：提出' + (r + 1) + 'は' + sum + '人'}"><title>ID ${label(mask)}${sum === null ? '' : '：提出' + (r + 1) + 'は' + sum + '人'}</title></path>`;
    }).join('') + layout.nodes.filter(node => cap === 2 && alive.has(node.id)).map(node => `<circle class="pair-node${stage === 'removing' && !nextAlive.has(node.id) ? ' fading' : ''}${unique ? ' unique' : ''}" cx="${node.x}" cy="${node.y}" r="3" data-node-id="${node.id}"></circle>`).join('');
    el('pair-stage').setAttribute('data-density', state.remainingMasks.length > 15 ? 'dense' : 'sparse');
    if (focusedPair !== null) {
      const selected = shared.maskIDs(focusedPair, ids), values = selected.map(id => Number(Boolean(current.query & 1 << ids.indexOf(id))));
      el('pair-detail').textContent = `ID ${label(focusedPair)}の線。${state.revealedCount ? `提出${r + 1}は ${values.join(' ＋ ')} ＝ ${values.reduce((n, bit) => n + bit, 0)}人。結果${current.hit}人と${pop(focusedPair & current.query) === current.hit ? '一致' : '不一致'}。` : '2人の組を1本の線で表します。'}`;
    } else el('pair-detail').textContent = cap === 1 ? (unique ? `ID ${label(state.remainingMasks[0])}の行だけが残りました。` : `${state.remainingMasks.length}人の行が、まだ所属の候補として残っています。`)
      : unique ? `ID ${label(state.remainingMasks[0])}を結ぶ1本だけが残りました。` : `${state.remainingMasks.length}本の線が、まだあり得る2人の組です。線を押すとその組を確認できます。`;
    el('summary-result').textContent = unique ? `残る1${unit()}：ID ${label(state.remainingMasks[0])}` : `${depth}回分で${state.remainingMasks.length}${unit()}が残っています`;
    el('count-path').textContent = state.counts.join(' → ');
    el('merge-progress').textContent = phase?.stage === 'result' && !playing ? `提出${depth + 1}：絞り込み待ち` : `${depth} / 5提出`;
    el('animate-all').textContent = playing ? '止める（手動に戻る）' : depth === 5 ? '最初から再生' : depth ? '続きを自動で表示' : '5回の結果を順に表示';
    el('animate-all').disabled = busy && !playing;
    el('merge-one').textContent = busy ? '絞り込み中…' : phase?.stage === 'result' ? 'この結果で候補を絞る' : '次の提出結果を表示';
    el('merge-one').disabled = playing || busy || depth === 5;
    el('unmerge-one').disabled = depth === 0 && !phase;
    el('spread-all').disabled = depth === 0 && !phase && !playing;
    el('current-calculations').innerHTML = (cap === 2 ? '<p>最初の候補：11×10÷2＝55組。</p>' : '<p>最初の候補：11人から1人を選ぶので11人。</p>') + rows.slice(0, depth).map(row => `<details><summary>提出${row.step + 1}：${row.before.length}${unit()} → ${row.after.length}${unit()}（${row.hit}人）</summary><p>対象ID：${label(row.query)}。前まで残った${cap === 2 ? '組' : '行'}を、この提出の人数で分けます。</p>${row.groups.map((group, hit) => `<p class="data-candidates"><b>${hit}人：${group.length}${unit()}</b> — ${group.length ? group.map(label).join(' ／ ') : 'なし'}</p>`).join('')}</details>`).join('');
    if (globalThis.MIA_FIXED_OVERVIEW) globalThis.MIA_FIXED_OVERVIEW.fromLesson({cap, member, depth, revealedCount: state.revealedCount});
  }
  function stop() {epoch++; playing = false; busy = false; phase = null;}
  function pause() {epoch++; playing = false; busy = false; if (phase) phase = {...phase, stage: 'result'};}
  function setAnswer(capacity, target, order) {
    if (![1, 2].includes(capacity) || !five.candidates(capacity).includes(target)) throw Error('所属人数分の異なるIDを選んでください。');
    const chosen = order || shared.maskIDs(target, ids);
    if (!Array.isArray(chosen) || chosen.length !== capacity || new Set(chosen).size !== capacity || chosen.some(id => !ids.includes(id)) || chosen.reduce((mask, id) => mask | 1 << ids.indexOf(id), 0) !== target) throw Error('所属人数分の異なるIDを選んでください。');
    stop(); cap = capacity; member = target; masks[cap] = member; pairOrder = chosen.slice(); depth = 0; focusedPair = null;
    el('lesson-error').hidden = true; render();
  }
  function setTab(capacity) {
    if (![1, 2].includes(capacity)) throw Error('所属1人か2人を選んでください。');
    if (globalThis.MIA_FIXED_OVERVIEW) globalThis.MIA_FIXED_OVERVIEW.selectCapacity(capacity);
    else setAnswer(capacity, masks[capacity]);
  }
  function setSpeed(value) {
    if (!Object.hasOwn(speeds, value)) throw Error('再生速度を選んでください。');
    pause(); speed = value; render();
  }
  function setDepth(value) {
    if (!Number.isInteger(value) || value < 0 || value > 5) throw Error('提出数は0〜5です。');
    stop(); depth = value; render();
  }
  function focusPair(mask) {
    if (cap !== 2 || mask !== null && !snapshot().displayedMasks.includes(mask)) throw Error('現在残る線を選んでください。');
    focusedPair = focusedPair === mask ? null : mask; render();
  }
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  function showResult() {
    phase = {round: depth, hit: pop(data.query_masks[depth] & member), stage: 'result'}; render();
  }
  async function applyResult(ticket) {
    if (depth >= 5 || ticket !== epoch) return;
    phase = {round: depth, hit: pop(data.query_masks[depth] & member), stage: 'removing'};
    busy = true; render(); await delay(timing().fade);
    if (ticket !== epoch) return;
    depth++; phase = null; busy = false; render();
  }
  async function revealNext(ticket) {
    if (depth >= 5 || ticket !== epoch) return;
    showResult(); await delay(timing().result);
    if (ticket !== epoch) return;
    await applyResult(ticket);
  }
  async function advanceOne() {
    if (playing || busy || depth >= 5) return;
    const ticket = ++epoch;
    if (!phase) {showResult(); return;}
    await applyResult(ticket);
  }
  async function animateAll() {
    if (playing) {pause(); render(); return;}
    if (busy) return;
    const ticket = ++epoch; playing = true;
    if (depth === 5) {depth = 0; phase = null; focusedPair = null; render(); await delay(200);}
    while (depth < 5 && ticket === epoch) {
      await revealNext(ticket); if (ticket !== epoch) return;
      await delay(timing().gap);
    }
    if (ticket === epoch) {playing = false; render();}
  }
  function chooseRecords() {
    try {
      const chosen = (cap === 2 ? [1, 2] : [1]).map(i => Number(el(`answer-record-${i}`).value));
      setAnswer(cap, chosen.reduce((mask, id) => mask | (ids.includes(id) ? 1 << ids.indexOf(id) : 0), 0), chosen);
    } catch (error) {el('lesson-error').textContent = error.message; el('lesson-error').hidden = false; pickers();}
  }
  el('tab-single').onclick = () => setTab(1); el('tab-pair').onclick = () => setTab(2);
  el('playback-speed').onchange = () => setSpeed(el('playback-speed').value);
  el('answer-record-1').onchange = chooseRecords; el('answer-record-2').onchange = chooseRecords;
  el('animate-all').onclick = animateAll; el('merge-one').onclick = advanceOne;
  el('unmerge-one').onclick = () => setDepth(phase ? depth : Math.max(0, depth - 1)); el('spread-all').onclick = () => setDepth(0);
  el('pair-lines').onclick = event => {const target = event.target.closest('[data-pair-mask]'); if (target) focusPair(Number(target.dataset.pairMask));};
  el('pair-lines').onkeydown = event => {if (event.key === 'Enter' || event.key === ' ') {const target = event.target.closest('[data-pair-mask]'); if (target) {event.preventDefault(); focusPair(Number(target.dataset.pairMask));}}};
  function help(open) {
    el('help-overlay').hidden = !open;
    const button = el(open ? 'close-help' : 'open-help'); if (typeof button.focus === 'function') button.focus();
  }
  el('open-help').onclick = () => help(true); el('close-help').onclick = () => help(false);
  el('help-overlay').onclick = event => {if (event.target === el('help-overlay')) help(false);};
  document.addEventListener('keydown', event => {
    if (el('help-overlay').hidden) return;
    if (event.key === 'Escape') help(false);
    if (event.key === 'Tab') {
      const focusable = [...el('help-panel').querySelectorAll('button,input,select,a[href],summary')].filter(node => !node.disabled && (!node.closest('details:not([open])') || node.tagName === 'SUMMARY'));
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {event.preventDefault(); last.focus();}
      else if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first.focus();}
    }
  });
  globalThis.MIA_FIXED_LESSON = Object.freeze({defaultMember, trace, remaining, geometry, snapshot, setAnswer, setTab, setSpeed, timing, setDepth, focusPair, advanceOne, animateOne: advanceOne, animateAll, stop, pause, render,
    values: () => ({cap, depth, masks: {...masks}, playing})});
  render();
})();
