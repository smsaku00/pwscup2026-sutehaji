/* Introductory, lessons using one fixed five-question code for every possible membership. */
(function () {
  'use strict';
  const colors = ['#2476b8', '#008576', '#bc6913', '#a4497a', '#6552ad'];
  const defaults = data.lesson_defaults;
  let depth = 5;
  const masks = {...defaults};
  const el = id => document.getElementById(id);
  const unit = cap => cap === 1 ? '候補' : '組';
  const label = mask => shared.maskIDs(mask, ids).join('・');
  const members = mask => shared.maskIDs(mask, ids);
  const all = cap => five.candidates(cap);

  function trace(cap, member) {
    if (!all(cap).includes(member)) throw Error('所属例の人数が一致しません。');
    let before = all(cap);
    return data.query_masks.map((query, step) => {
      const hit = pop(query & member);
      const grouped = five.groups(before, query);
      const after = before.filter(candidate => pop(candidate & query) === hit);
      const row = {step, query, hit, complete: false, alreadyUnique: before.length === 1,
        before: before.slice(), after: after.slice(),
        groups: Array.from({length: cap + 1}, (_, h) => grouped.get(h) || [])};
      before = after;
      return row;
    });
  }
  function remaining(cap, member, count) {
    const rows = trace(cap, member);
    return all(cap).filter(candidate => rows.slice(0, count).every(row => pop(candidate & row.query) === row.hit));
  }
  function badges(candidate, rows, count) {
    return rows.map((row, i) => {
      const pass = pop(candidate & row.query) === row.hit;
      return `<i class="condition ${i < count ? pass ? 'agree' : 'reject' : 'unused'}" style="--sheet:${colors[i]}" title="質問${i + 1}：${row.complete ? '追加質問不要' : pass ? '回答に一致' : '回答に不一致'}">${i + 1}</i>`;
    }).join('');
  }
  function card(candidate, cap, rows, count, sheet) {
    const pass = sheet === undefined
      ? rows.slice(0, count).every(row => pop(candidate & row.query) === row.hit)
      : pop(candidate & rows[sheet].query) === rows[sheet].hit;
    const signature = rows.map(row => pop(candidate & row.query) === row.hit ? '1' : '0').join('');
    return `<div class="lesson-member ${pass ? 'fits' : 'blocked'}" data-member="${candidate}" data-pass="${pass ? 1 : 0}" data-signature="${signature}" title="所属候補：${label(candidate)}。${pass ? '回答に一致' : '回答に不一致'}"><span>${members(candidate).join('<br>')}</span>${sheet === undefined ? `<div class="conditions">${badges(candidate, rows, count)}</div>` : ''}</div>`;
  }
  function question(row, cap, rows) {
    const i = row.step, color = colors[i];
    const selected = shared.maskIDs(row.query, ids);
    const cells = ids.map((id, j) => `<span class="id-position ${row.query >> j & 1 ? 'investigate' : ''}" data-id="${id}" data-selected="${row.query >> j & 1}" style="--sheet:${color}">${id}</span>`).join('');
    const counts = row.groups.map((group, h) => `<span class="${h === row.hit ? 'chosen-hit' : ''}">${h}人 → ${group.length}${unit(cap)}</span>`).join('');
    const enumeration = row.groups.map((group, h) => `<p><b>${h}人の回答に合う${group.length}${unit(cap)}：</b>${group.length ? group.map(label).join(' ／ ') : 'なし'}</p>`).join('');
    let calculation = '';
    if (i === 0 && cap === 1) calculation = '選択した5人 → 1人の回答なら5候補。選ばない6人 → 0人なら6候補。';
    if (i === 0 && cap === 2) calculation = '0人：6×5÷2＝15組 ／ 1人：5×6＝30組 ／ 2人：5×4÷2＝10組。合計55組。';
    return `<article class="question-sheet ${i < depth ? 'stacked' : 'not-stacked'} ${row.complete ? 'finished-sheet' : ''}" data-cap="${cap}" data-step="${i + 1}" data-query="${row.query}" data-hit="${row.hit}" data-before="${row.before.length}" data-after="${row.after.length}" style="--sheet:${color}">
      <div class="question-head"><b class="question-number">質問${i + 1}</b><span>${row.complete ? '確定済み・追加質問不要' : `この例の回答：<b>${row.hit}人</b>`}</span></div>
      <p class="question-count"><b>${row.before.length}${unit(cap)} → ${row.after.length}${unit(cap)}</b>${row.complete ? '（変わらない）' : ''}</p>
      <div class="id-strip" aria-label="質問${i + 1}で調べるID">${cells}</div>
      <p class="small">${row.complete ? 'すでに1枚に確定。この行では誰も調べません。' : `色の位置を調べる：<b>${selected.join('・')}</b>`}</p>
      ${row.complete ? '' : `<div class="hit-counts">${counts}</div><p class="small count-reason">${calculation || '前の質問まで残ったカードから、この回答に合うものを数えます。'}</p>`}
      <div class="candidate-sheet" aria-label="質問${i + 1}の条件だけを示す候補カード">${all(cap).map(candidate => card(candidate, cap, rows, depth, i)).join('')}</div>
      <p class="small muted">${row.complete ? '透明なままのシートです。' : '色のカードはこの回だけの回答に一致。灰色は不一致。上の候補数には前の回答も適用します。'}</p>
      ${row.complete ? '' : `<details class="enumeration"><summary>候補数の計算の元を見る</summary><div>${enumeration}</div></details>`}
    </article>`;
  }
  function renderColumn(cap) {
    const rows = trace(cap, masks[cap]);
    el(`lesson-counts-${cap}`).textContent = [all(cap).length, ...rows.map(row => row.after.length)].join(' → ');
    el(`lesson-questions-${cap}`).innerHTML = rows.map(row => question(row, cap, rows)).join('');
    const kept = remaining(cap, masks[cap], depth);
    el(`lesson-result-${cap}`).innerHTML = `<div class="merge-head"><h3>${depth}枚を重ねた候補カード</h3><p><b>${all(cap).length}${unit(cap)} → ${kept.length}${unit(cap)}</b></p></div>
      <p class="small">${depth ? `質問1〜${depth}の回答にすべて合うカードが緑です。どれか1枚で灰色になるカードは除外します。` : 'まだ条件を重ねていないので、すべての候補が残ります。'}</p>
      <div class="candidate-sheet combined-sheet" data-cap="${cap}" data-depth="${depth}" data-remaining="${kept.length}" aria-label="回答を重ねた最終候補カード">${all(cap).map(candidate => card(candidate, cap, rows, depth)).join('')}</div>
      <p class="small">カード下の1〜5の色は、その回の回答に合うかを示します。まだ重ねない回は白、合わない回は灰色です。</p>
      <div class="merge-answer">${kept.length === 1 ? `<b>残る1枚：ID ${label(kept[0])}</b><p class="small">${cap === 2 ? '2人の位置に印のある1枚のカードです。' : '1人の位置に印のある1枚のカードです。'}</p>` : `<b>${kept.length}${unit(cap)}がまだ残っています</b><p class="small">次のシートを重ねて絞り込みます。</p>`}</div>`;
  }
  function render() {
    [1, 2].forEach(renderColumn);
    el('lesson-depth').value = String(depth);
    el('lesson-step-label').textContent = depth ? `質問1〜${depth}の条件を重ねています` : '開始：まだ質問を重ねていません';
    el('lesson-back').disabled = depth === 0;
    el('lesson-next').disabled = depth === 5;
  }
  [1, 2].forEach(cap => {
    const select = el(`lesson-example-${cap}`);
    select.innerHTML = all(cap).map(candidate => `<option value="${candidate}"${candidate === masks[cap] ? ' selected' : ''}>ID ${label(candidate)}</option>`).join('');
    select.value = String(masks[cap]);
    select.onchange = () => {masks[cap] = Number(select.value); render();};
  });
  el('lesson-depth').onchange = () => {depth = Number(el('lesson-depth').value); render();};
  el('lesson-back').onclick = () => {depth = Math.max(0, depth - 1); render();};
  el('lesson-next').onclick = () => {depth = Math.min(5, depth + 1); render();};
  el('lesson-start').onclick = () => {depth = 0; render();};
  el('lesson-end').onclick = () => {depth = 5; render();};
  globalThis.MIA_FIXED_LESSON = Object.freeze({trace, remaining, render, values: () => ({depth, masks: {...masks}})});
  render();
})();
