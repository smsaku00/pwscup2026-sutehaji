/* Eleven record patterns; add one or two selected records and compare with a Team. */
(function () {
  'use strict';
  const el = id => document.getElementById(id);
  const all = capacity => five.candidates(capacity);
  const label = mask => shared.maskIDs(mask, ids).join('・');
  const unit = capacity => capacity === 1 ? '候補' : '組';
  const pattern = id => data.query_masks.map(query => Number(Boolean(query & (1 << ids.indexOf(id)))));
  const masks = {...data.lesson_defaults};
  let pairOrder = shared.maskIDs(masks[2], ids);
  let cap = 1, depth = 0, playing = false, epoch = 0, merged = false, addingRound = -1;
  let trialIDs = shared.maskIDs(masks[cap], ids);
  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function trace(capacity, member) {
    if (!all(capacity).includes(member)) throw Error('正解レコードの人数が一致しません。');
    let before = all(capacity);
    return data.query_masks.map((query, step) => {
      const hit = pop(query & member), grouped = five.groups(before, query);
      const after = before.filter(candidate => pop(candidate & query) === hit);
      const row = {step, query, hit, complete: false, alreadyUnique: before.length === 1,
        before: before.slice(), after: after.slice(),
        groups: Array.from({length: capacity + 1}, (_, h) => grouped.get(h) || [])};
      before = after;
      return row;
    });
  }
  function remaining(capacity, member, count) {
    const rows = trace(capacity, member);
    return all(capacity).filter(candidate => rows.slice(0, count).every(row => pop(candidate & row.query) === row.hit));
  }
  function trialSnapshot() {
    const signatures = trialIDs.map(pattern);
    const sum = data.query_masks.map((_, r) => signatures.reduce((n, bits) => n + bits[r], 0));
    const target = data.query_masks.map(query => pop(query & masks[cap]));
    const complete = trialIDs.length === cap, matches = sum.map((n, r) => n === target[r]);
    return {selectedIDs: trialIDs.slice(), signatures, sum, target, complete, matches,
      allMatch: complete && matches.every(Boolean), merged, comparedCount: depth, addingRound};
  }
  function geometry(height) {
    const h = Math.max(0, Number(height) || 0), gap = Math.min(h / 20, 4), cardHeight = (h - 2 * gap) / 3;
    return {height: h, gap, cardHeight, positions: [0, cardHeight + gap, 2 * (cardHeight + gap)], target: 2 * (cardHeight + gap)};
  }
  function place(reflow = false) {
    const layout = geometry(el('overlay-stage').clientHeight || 108);
    for (let i = 0; i < 2; i++) {
      const card = el(`floating-record-${i + 1}`);
      card.hidden = !trialIDs[i];
      card.classList.toggle('reflow', reflow); card.classList.toggle('adding', addingRound >= 0);
      card.style.setProperty('--addition-distance', `${layout.target - layout.positions[i]}px`);
      card.style.setProperty('--addition-offset', `${i === 0 ? -9 : 9}px`);
      card.style.height = `${layout.cardHeight}px`;
      card.style.transform = `translateY(${layout.positions[i]}px)`;
      card.setAttribute('data-adding-round', String(addingRound));
    }
    const result = el('sum-result');
    result.classList.toggle('reflow', reflow); result.style.height = `${layout.cardHeight}px`;
    result.style.transform = `translateY(${layout.target}px)`;
    if (reflow && typeof requestAnimationFrame === 'function') requestAnimationFrame(() => {
      for (let i = 1; i <= 2; i++) el(`floating-record-${i}`).classList.remove('reflow');
      result.classList.remove('reflow');
    });
  }
  function pickers() {
    const selected = cap === 2 ? pairOrder : shared.maskIDs(masks[cap], ids);
    [1, 2].forEach((number, i) => {
      const select = el(`answer-record-${number}`);
      select.innerHTML = ids.map(id => `<option value="${id}"${id === selected[i] ? ' selected' : ''}${cap === 2 && id === selected[1 - i] ? ' disabled' : ''}>ID ${id}</option>`).join('');
      select.value = String(selected[i] === undefined ? ids[1] : selected[i]);
    });
    el('answer-record-2-wrap').hidden = cap !== 2;
  }
  function render() {
    const rows = trace(cap, masks[cap]), kept = remaining(cap, masks[cap], depth), state = trialSnapshot();
    const solved = depth === 5 && state.allMatch;
    [1, 2].forEach(c => el(c === 1 ? 'tab-single' : 'tab-pair').setAttribute('aria-selected', String(c === cap)));
    pickers();
    el('record-deck').innerHTML = ids.map(id => {
      const selected = trialIDs.includes(id), bits = pattern(id);
      return `<button type="button" class="record-card${selected ? ' selected' : ''}${selected && solved ? ' solved' : ''}" data-record-id="${id}" data-signature="${bits.join('')}" aria-pressed="${selected}" aria-label="ID ${id}：提出1〜5の模様 ${bits.join('')}。${selected ? '比較から外す' : '比較に選ぶ'}"><b>ID ${id}</b>${bits.map((n, r) => `<span class="record-bit bit-${n}" data-round="${r}" data-value="${n}" title="提出${r + 1}の対象${n ? '内' : '外'}">${n}</span>`).join('')}</button>`;
    }).join('');
    el('trial-hint').textContent = `${cap}人を選ぶ → ${cap === 2 ? '各桁を足す' : 'その模様'} → 上の答えと比べる`;
    for (let i = 0; i < 2; i++) {
      const id = trialIDs[i], card = el(`floating-record-${i + 1}`);
      card.innerHTML = id === undefined ? '' : `<b>ID ${id}</b>${pattern(id).map((n, r) => `<span class="floating-bit bit-${n}${r === addingRound ? ' adding-digit' : ''}" data-round="${r}">${n}</span>`).join('')}`;
      card.setAttribute('aria-label', id === undefined ? '未選択' : `選んだID ${id}の模様 ${pattern(id).join('')}`);
    }
    el('sum-result').innerHTML = '<b>合計</b>' + state.sum.map((n, r) => {
      const checked = state.complete && r < depth, okay = checked && state.matches[r];
      return `<span class="sum-bit ${checked ? okay ? 'match' : 'mismatch' : r === addingRound ? 'calculating' : 'waiting'}" data-round="${r}" data-value="${n}" data-checked="${checked ? 1 : 0}" data-match="${checked ? okay ? 1 : 0 : ''}">${checked ? n : '·'}<small>${checked ? okay ? '✓' : '×' : ''}</small></span>`;
    }).join('');
    el('target-row').innerHTML = '<b>答え<br>縦5桁</b>' + state.target.map((n, r) => `<span class="target-bit${r === addingRound ? ' active' : ''}" data-round="${r}" data-value="${n}">${n}</span>`).join('');
    const equation = state.signatures.map(bits => bits.join('')).join(' ＋ ');
    if (!state.complete) el('trial-status').textContent = `あと${cap - trialIDs.length}人選んでください。チームの結果は変わりません。`;
    else if (solved) el('trial-status').textContent = `${equation} ＝ ${state.sum.join('')}。5回とも一致 → ID ${trialIDs.join('＋')}`;
    else if (depth === 5) el('trial-status').textContent = `${equation} ＝ ${state.sum.join('')}。提出${state.matches.map((okay, r) => okay ? null : r + 1).filter(Boolean).join('・')}が不一致です。別の${cap === 2 ? '2人' : '1人'}を選べます。`;
    else if (addingRound >= 0) el('trial-status').textContent = `提出${addingRound + 1}：${state.signatures.map(bits => bits[addingRound]).join(' ＋ ')} → 合計を計算中`;
    else if (depth) el('trial-status').textContent = `提出${depth}：${state.signatures.map(bits => bits[depth - 1]).join(' ＋ ')} ＝ ${state.sum[depth - 1]}。答え${state.target[depth - 1]}と${state.matches[depth - 1] ? '一致' : '不一致'}。`;
    else el('trial-status').textContent = '1提出ずつ照合すると、その提出の数字が合計欄へ移動します。';
    el('trial-status').classList.toggle('solved', solved);
    el('merge-progress').textContent = `${depth} / 5提出`;
    el('animate-all').textContent = playing ? '止める' : depth === 5 ? 'もう一度重ねる' : '選んだ模様を重ねる';
    el('animate-all').disabled = !state.complete;
    el('merge-one').disabled = !state.complete || depth === 5 || playing;
    el('unmerge-one').disabled = depth === 0;
    el('spread-all').disabled = depth === 0 && !merged && !playing;
    el('summary-result').textContent = `${kept.length}${unit(cap)}が人数列に一致`;
    el('summary-note').textContent = `11枚は各人の模様。${cap === 2 ? '55は2人の組み合わせの数です。' : '1人の模様を、そのまま比べます。'}`;
    el('count-path').textContent = [all(cap).length, ...rows.map(row => row.after.length)].join(' → ');
    el('current-calculations').innerHTML = `<p>説明例の所属：<b>ID ${label(masks[cap])}</b>。最初は${all(cap).length}${unit(cap)}です。</p>` + rows.map(row => `<details><summary>提出${row.step + 1}：${row.before.length}${unit(cap)} → ${row.after.length}${unit(cap)}（${row.hit}人）</summary><p>前の提出まで残った候補を、提出の対象集合に何人入るかで分けます。対象ID：${label(row.query)}</p>${row.groups.map((group, hit) => `<p class="data-candidates"><b>${hit}人：${group.length}${unit(cap)}</b> — ${group.length ? group.map(label).join(' ／ ') : 'なし'}</p>`).join('')}</details>`).join('');
    place();
    if (globalThis.MIA_FIXED_OVERVIEW) {globalThis.MIA_FIXED_OVERVIEW.fromLesson({cap, member: masks[cap], depth});
      if (addingRound >= 0) globalThis.MIA_FIXED_OVERVIEW.selectRound(addingRound);}
  }
  function stop() {epoch++; playing = false; addingRound = -1;}
  function resetComparison() {depth = 0; merged = false;}
  function setTrial(selected) {
    if (!Array.isArray(selected) || selected.length > cap || new Set(selected).size !== selected.length || selected.some(id => !ids.includes(id))) throw Error('人数分まで、異なる候補IDを選んでください。');
    stop(); trialIDs = selected.slice(); resetComparison(); render(); place(true);
  }
  function chooseTrial(id) {
    const next = trialIDs.slice(), index = next.indexOf(id);
    if (index >= 0) next.splice(index, 1);
    else {if (next.length === cap) next.shift(); next.push(id);}
    setTrial(next);
  }
  function findMatching() {
    const hits = trialSnapshot().target;
    const matches = all(cap).filter(member => data.query_masks.every((query, r) => pop(member & query) === hits[r]));
    if (matches.length !== 1) throw Error('人数列に一致する所属が一意に決まりません。');
    setTrial(shared.maskIDs(matches[0], ids));
  }
  function setTab(value) {
    if (![1, 2].includes(value)) throw Error('人数は1人か2人です。');
    stop(); cap = value; resetComparison(); trialIDs = shared.maskIDs(masks[cap], ids);
    el('lesson-error').hidden = true; render(); place(true);
  }
  function setAnswer(capacity, member, order) {
    if (!all(capacity).includes(member)) throw Error('異なる正解レコードを人数分選んでください。');
    if (capacity === 2) pairOrder = order || shared.maskIDs(member, ids);
    stop(); masks[capacity] = member; cap = capacity; resetComparison(); trialIDs = shared.maskIDs(member, ids);
    el('lesson-error').hidden = true; render(); place(true);
  }
  function setDepth(value) {
    if (!Number.isInteger(value) || value < 0 || value > 5) throw Error('照合する提出数は0〜5です。');
    if (value && !trialSnapshot().complete) throw Error('比較するレコードを人数分選んでください。');
    stop(); depth = value; merged = value > 0; render();
  }
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  async function addNext(ticket) {
    if (depth >= 5 || ticket !== epoch) return;
    addingRound = depth; render();
    await delay(reduced ? 20 : 680);
    if (ticket !== epoch) return;
    depth++; merged = true; addingRound = -1; render();
  }
  async function animateOne() {
    if (playing || depth >= 5 || !trialSnapshot().complete) return;
    stop(); const ticket = epoch; playing = true;
    await addNext(ticket);
    if (ticket === epoch) {playing = false; render();}
  }
  async function animateAll() {
    if (playing) {stop(); render(); return;}
    if (!trialSnapshot().complete) return;
    stop(); const ticket = epoch; playing = true;
    if (depth === 5) {resetComparison(); render(); await delay(reduced ? 20 : 200);}
    if (ticket !== epoch) return;
    while (depth < 5 && ticket === epoch) {
      await addNext(ticket);
      if (ticket !== epoch) return;
      await delay(reduced ? 20 : 180);
    }
    if (ticket === epoch) {playing = false; render();}
  }
  function chooseRecords() {
    try {
      const one = ids.indexOf(Number(el('answer-record-1').value)), two = ids.indexOf(Number(el('answer-record-2').value));
      if (one < 0 || cap === 2 && (two < 0 || one === two)) throw Error('2人のときは異なるIDを選んでください。');
      setAnswer(cap, (1 << one) | (cap === 2 ? 1 << two : 0), cap === 2 ? [ids[one], ids[two]] : undefined);
    } catch (error) {el('lesson-error').textContent = error.message; el('lesson-error').hidden = false; pickers();}
  }
  el('record-deck').onclick = event => {const target = event.target.closest('[data-record-id]'); if (target) chooseTrial(Number(target.dataset.recordId));};
  el('find-matching').onclick = findMatching;
  el('tab-single').onclick = () => setTab(1); el('tab-pair').onclick = () => setTab(2);
  el('answer-record-1').onchange = chooseRecords; el('answer-record-2').onchange = chooseRecords;
  el('animate-all').onclick = animateAll; el('merge-one').onclick = animateOne;
  el('unmerge-one').onclick = () => setDepth(Math.max(0, depth - 1)); el('spread-all').onclick = () => setDepth(0);
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
  if (typeof ResizeObserver === 'function') new ResizeObserver(() => place(true)).observe(el('overlay-stage'));
  else if (typeof addEventListener === 'function') addEventListener('resize', () => place(true));
  globalThis.MIA_FIXED_LESSON = Object.freeze({trace, remaining, geometry, setTab, setAnswer, setTrial, trialSnapshot, setDepth, animateOne, animateAll, stop, render, place,
    values: () => ({cap, depth, masks: {...masks}, playing})});
  render(); place(true);
})();
