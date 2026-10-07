/* Compact lesson: the same five investigation sets, with movable transparent sheets. */
(function () {
  'use strict';
  const el = id => document.getElementById(id);
  const all = cap => five.candidates(cap);
  const label = mask => shared.maskIDs(mask, ids).join('・');
  const unit = cap => cap === 1 ? '候補' : '組';
  const masks = {...data.lesson_defaults};
  let pairOrder = shared.maskIDs(masks[2], ids);
  let cap = 1, depth = 0, playing = false, epoch = 0;
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
  function geometry(height) {
    const h = Math.max(0, Number(height) || 0), gap = Math.min(h / 30, h < 350 ? 4 : 6);
    const cardHeight = (h - gap * 5) / 6;
    return {height: h, gap, cardHeight, positions: Array.from({length: 6}, (_, i) => i * (cardHeight + gap)), target: 5 * (cardHeight + gap)};
  }
  function place(reflow = false) {
    const layout = geometry(el('card-stage').clientHeight || 400);
    for (let i = 0; i < 5; i++) {
      const card = el(`glass-card-${i + 1}`);
      card.classList.toggle('reflow', reflow);
      card.classList.toggle('merged', i < depth);
      card.style.height = `${layout.cardHeight}px`;
      card.style.transform = `translateY(${i < depth ? layout.target : layout.positions[i]}px)`;
      card.setAttribute('data-merged', i < depth ? '1' : '0');
    }
    const base = el('merge-base');
    base.classList.toggle('reflow', reflow);
    base.style.height = `${layout.cardHeight}px`;
    base.style.transform = `translateY(${layout.target}px)`;
    if (reflow && typeof requestAnimationFrame === 'function') requestAnimationFrame(() => {
      for (let i = 1; i <= 5; i++) el(`glass-card-${i}`).classList.remove('reflow');
      base.classList.remove('reflow');
    });
  }
  function cells(rows, sheet) {
    return all(cap).map(candidate => {
      const pass = sheet === undefined ? rows.slice(0, depth).every(row => pop(candidate & row.query) === row.hit)
        : pop(candidate & rows[sheet].query) === rows[sheet].hit;
      return `<div class="sheet-cell ${pass ? 'fits' : 'blocked'}${candidate === masks[cap] ? ' current' : ''}" data-member="${candidate}" data-pass="${pass ? 1 : 0}" title="ID ${label(candidate)}：${pass ? '人数に一致' : '人数に不一致'}"><span class="cell-label">${shared.maskIDs(candidate, ids).join('＋')}</span></div>`;
    }).join('');
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
    const rows = trace(cap, masks[cap]), kept = remaining(cap, masks[cap], depth);
    [1, 2].forEach(c => {el(c === 1 ? 'tab-single' : 'tab-pair').setAttribute('aria-selected', String(c === cap));});
    pickers();
    rows.forEach((row, i) => {
      const card = el(`glass-card-${i + 1}`), grid = el(`glass-grid-${i + 1}`);
      card.setAttribute('data-query', String(row.query));
      card.setAttribute('data-hit', String(row.hit));
      card.setAttribute('aria-label', `提出${i + 1}：ID ${label(row.query)}を対象にする。この例は${row.hit}人。`);
      el(`glass-meta-${i + 1}`).innerHTML = `<strong>提出${i + 1}</strong><span class="investigation-ids">対象ID：${label(row.query)}</span><span class="hit">${row.hit}人</span><span class="investigation-ids">${row.before.length} → ${row.after.length}</span>`;
      grid.style.setProperty('--rows', cap === 1 ? '1' : '5');
      grid.setAttribute('data-cap', String(cap));
      grid.innerHTML = cells(rows, i);
    });
    const grid = el('merge-grid');
    grid.style.setProperty('--rows', cap === 1 ? '1' : '5');
    grid.setAttribute('data-cap', String(cap));
    grid.innerHTML = cells(rows);
    el('merge-base-title').textContent = depth ? `${depth}枚を重ねた色` : 'ここに重ねる';
    el('merge-base-count').textContent = `残る ${kept.length}${unit(cap)}`;
    el('merge-progress').textContent = `${depth} / 5枚`;
    el('animate-all').textContent = playing ? '止める' : depth === 5 ? 'もう一度重ねる' : '順に重ねる';
    el('merge-one').disabled = depth === 5;
    el('unmerge-one').disabled = depth === 0;
    el('spread-all').disabled = depth === 0 && !playing;
    el('summary-result').textContent = kept.length === 1 ? `残る1枚：ID ${label(kept[0])}` : `${kept.length}${unit(cap)}が残っています`;
    el('summary-note').textContent = depth === 0 ? '各カードはその提出だけの条件。重ねると、すべてに合う位置が残ります。' : depth === 5 ? '5回とも人数が一致する位置が緑です。正解レコードを変えて試せます。' : `提出1〜${depth}の人数にすべて合う位置が緑です。`;
    el('count-path').textContent = [all(cap).length, ...rows.map(row => row.after.length)].join(' → ');
    el('current-calculations').innerHTML = `<p>選んだ正解：<b>ID ${label(masks[cap])}</b>。最初は${all(cap).length}${unit(cap)}です。</p>` + rows.map(row => `<details><summary>提出${row.step + 1}：${row.before.length}${unit(cap)} → ${row.after.length}${unit(cap)}（${row.hit}人）</summary><p>前の提出まで残った候補を、提出の対象集合に何人入るかで分けます。</p>${row.groups.map((group, hit) => `<p class="data-candidates"><b>${hit}人：${group.length}${unit(cap)}</b> — ${group.length ? group.map(label).join(' ／ ') : 'なし'}</p>`).join('')}</details>`).join('');
    place();
    if (globalThis.MIA_FIXED_OVERVIEW) globalThis.MIA_FIXED_OVERVIEW.fromLesson({cap, member: masks[cap], depth});
  }
  function stop() {epoch++; playing = false;}
  function setTab(value) {
    if (![1, 2].includes(value)) throw Error('人数は1人か2人です。');
    stop(); cap = value; depth = 0; el('lesson-error').hidden = true; render(); place(true);
  }
  function setAnswer(capacity, member, order) {
    if (!all(capacity).includes(member)) throw Error('異なる正解レコードを人数分選んでください。');
    if (capacity === 2) pairOrder = order || shared.maskIDs(member, ids);
    stop(); masks[capacity] = member; cap = capacity; depth = 0; el('lesson-error').hidden = true; render(); place(true);
  }
  function setDepth(value) {
    if (!Number.isInteger(value) || value < 0 || value > 5) throw Error('重ねる枚数は0〜5です。');
    stop(); depth = value; render();
  }
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  async function animateAll() {
    if (playing) {stop(); render(); return;}
    stop(); const ticket = epoch; playing = true;
    if (depth === 5) {depth = 0; render(); await delay(reduced ? 20 : 920);}
    if (ticket !== epoch) return;
    render();
    while (depth < 5 && ticket === epoch) {
      depth++; render(); await delay(reduced ? 90 : 1020);
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
  el('tab-single').onclick = () => setTab(1);
  el('tab-pair').onclick = () => setTab(2);
  el('answer-record-1').onchange = chooseRecords;
  el('answer-record-2').onchange = chooseRecords;
  el('animate-all').onclick = animateAll;
  el('merge-one').onclick = () => setDepth(Math.min(5, depth + 1));
  el('unmerge-one').onclick = () => setDepth(Math.max(0, depth - 1));
  el('spread-all').onclick = () => setDepth(0);
  function help(open) {
    el('help-overlay').hidden = !open;
    const button = el(open ? 'close-help' : 'open-help');
    if (typeof button.focus === 'function') button.focus();
  }
  el('open-help').onclick = () => help(true);
  el('close-help').onclick = () => help(false);
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
  if (typeof ResizeObserver === 'function') new ResizeObserver(() => place(true)).observe(el('card-stage'));
  else if (typeof addEventListener === 'function') addEventListener('resize', () => place(true));
  globalThis.MIA_FIXED_LESSON = Object.freeze({trace, remaining, geometry, setTab, setAnswer, setDepth, animateAll, stop, render, place,
    values: () => ({cap, depth, masks: {...masks}, playing})});
  render(); place(true);
})();
