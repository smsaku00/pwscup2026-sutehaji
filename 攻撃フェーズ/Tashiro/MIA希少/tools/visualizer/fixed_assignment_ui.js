/* Read one record horizontally, then gather the included records vertically. */
(function () {
  'use strict';
  const el = id => document.getElementById(id), placed = new Set();
  let selected = 0, playing = false, epoch = 0, moving = null;
  const bits = i => data.query_masks.map(query => query >> i & 1);
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  function snapshot() {
    return {selected, selectedID: ids[selected], pattern: bits(selected), placed: [...placed], playing, moving,
      destinations: data.query_masks.map(query => ids.filter((_, i) => placed.has(i) && (query >> i & 1)))};
  }
  function render() {
    const pattern = bits(selected), complete = placed.size === ids.length;
    el('assignment-record').innerHTML = ids.map((id, i) => `<option value="${id}"${i === selected ? ' selected' : ''}>ID ${id}</option>`).join('');
    el('assignment-record').value = String(ids[selected]);
    el('assignment-focus').innerHTML = `<div class="assignment-source"><span>選んだレコード</span><b>ID ${ids[selected]}</b><small>同じIDを、含める回へ ↓</small></div>` + pattern.map((bit, r) => `<div class="assignment-decision ${bit ? 'include' : 'exclude'}" data-assignment-decision="${r}" data-bit="${bit}"><span>提出${r + 1}</span><b>${bit ? '● 含める' : '○ 含めない'}</b><small>${bit} ${bit ? '→ 対象に入れる' : '→ 対象外にする'}</small></div>`).join('');
    el('assignment-grid').innerHTML = '<div class="assignment-grid-head"><b>レコード ↓</b>' + data.query_masks.map((_, r) => `<b>提出${r + 1} ↓</b>`).join('') + '</div>' + ids.map((id, i) => `<div class="assignment-row${i === selected ? ' selected' : ''}${placed.has(i) ? ' placed' : ' waiting'}" data-assignment-row="${i}"><button data-assignment-record="${i}" aria-pressed="${i === selected}" aria-label="ID ${id}の5回の振り分けを見る">${id}${placed.has(i) ? ' ✓' : ''}</button>` + bits(i).map((bit, r) => `<span data-assignment-cell="${i}:${r}" data-bit="${bit}" class="${bit ? 'include' : 'exclude'}">${bit ? '●' : '○'}<small>${bit}</small></span>`).join('') + '</div>').join('');
    const travel = Math.max(20, el('assignment-grid').clientHeight * (1 - (selected + .5) / ids.length) + 18);
    el('assignment-destinations').innerHTML = '<div class="assignment-destination-label"><b>列を下に集める ↓</b><span>● のIDだけが<br>その提出の対象</span></div>' + data.query_masks.map((query, r) => {
      const members = ids.map((id, i) => ({id, i})).filter(x => placed.has(x.i) && (query >> x.i & 1));
      return `<div class="assignment-destination${pattern[r] ? ' selected-column' : ''}" data-assignment-destination="${r}"><b>提出${r + 1}の対象 <small>${members.length}/5</small></b><div>${members.map(x => `<span data-destination-id="${x.id}" class="assignment-id${moving === x.i ? ' arriving' : ''}" style="--assignment-travel:${travel}px">${x.id}</span>`).join('')}${members.length ? '' : '<span class="assignment-empty">まだ空</span>'}</div></div>`;
    }).join('');
    el('assignment-step').textContent = placed.has(selected) ? (complete ? '11レコードの振り分け完了' : '次のレコードを見る') : `ID ${ids[selected]}を振り分ける`;
    el('assignment-step').disabled = playing || complete;
    el('assignment-play').textContent = playing ? '止める（手動に戻る）' : '11レコードを順に再生';
    el('assignment-all').disabled = playing || complete;
    el('assignment-status').textContent = `${placed.size} / 11レコードを振り分け済み`;
    el('assignment-instruction').textContent = complete ? '11行の「含める」を列ごとに集めると、5回分の対象IDになります。ここまでを、結果を見る前に固定します。' : `ID ${ids[selected]}は提出${pattern.map((bit, r) => bit ? r + 1 : null).filter(n => n !== null).join('・')}に含めます。横に5回分を決め、列ごとに対象IDへ集めます。`;
  }
  function stop() {epoch++; playing = false; moving = null; render();}
  function select(index) {
    if (!Number.isInteger(index) || index < 0 || index >= ids.length) throw Error('レコードを選んでください。');
    stop(); selected = index; render();
  }
  function place() {placed.add(selected); moving = selected; render();}
  function step() {
    if (playing || placed.size === ids.length) return;
    epoch++;
    if (placed.has(selected)) {selected = ids.findIndex((_, i) => !placed.has(i)); moving = null; render();}
    else place();
  }
  function showAll() {stop(); ids.forEach((_, i) => placed.add(i)); render();}
  function reset() {stop(); placed.clear(); selected = 0; render();}
  async function play() {
    if (playing) {stop(); return;}
    if (placed.size === ids.length) {placed.clear(); selected = 0;}
    const ticket = ++epoch; playing = true; moving = null; render();
    while (placed.size < ids.length && ticket === epoch) {
      if (placed.has(selected)) selected = ids.findIndex((_, i) => !placed.has(i));
      moving = null; render(); await wait(1600); if (ticket !== epoch) return;
      place(); await wait(reduced() ? 1600 : 1800); if (ticket !== epoch) return;
    }
    playing = false; moving = null; render();
  }
  el('assignment-record').onchange = () => select(ids.indexOf(Number(el('assignment-record').value)));
  el('assignment-grid').onclick = event => {const target = event.target.closest('[data-assignment-record]'); if (target) select(Number(target.dataset.assignmentRecord));};
  el('assignment-step').onclick = step; el('assignment-play').onclick = play;
  el('assignment-all').onclick = showAll; el('assignment-reset').onclick = reset;
  el('assignment-next').onclick = () => globalThis.MIA_FIXED_OVERVIEW.setView('overview');
  globalThis.MIA_FIXED_ASSIGNMENT = Object.freeze({snapshot, select, step, play, stop, reset, showAll, render});
  render();
})();
