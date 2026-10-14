'use strict';

// ---------------------------------------------------------------- strings
const I18N = {
  zh: {
    addFiles: '＋ 檔案', addFolder: '＋ 資料夾', model: '模型', stop: '■ 停止',
    runAll: n => `▶ 分割待處理 (${n})`, runChecked: n => `▶ 分割已選 (${n})`,
    settings: '設定', pathsLabel: '檔案或資料夾路徑（每行一個）',
    include: '包含關鍵字（逗號分隔，空白=全部）', exclude: '排除關鍵字', add: '加入', cancel: '取消',
    saveMat: '在工作資料夾輸出 .mat（舊版格式）', matInput: '.mat 也存到原始資料夾',
    niiInput: '分割 .nii.gz 存到原始資料夾', workspace: '工作資料夾', openFolder: '開啟資料夾',
    cases: '病例', colName: '名稱', colStatus: '狀態', colApex: 'Apex', colChecks: '檢查',
    emptyCases: '按「＋ 檔案」或「＋ 資料夾」加入 cine NIfTI。\n不需要先準備 CSV，也不用先填 Apex：分割完成後再看著影像確認。',
    remove: '移除選取', openCaseFolder: '開啟病例資料夾',
    viewSingle: '單張', viewMosaic: '全部 slice', zoomHeart: '放大心臟', zoomFull: '整張影像',
    ovSeg: '分割', ovEdge: '輪廓', ovAha: 'AHA 17', ovNone: '無疊圖',
    opacity: '透明度', hintSelect: '從左側選一個病例', frame: 'Frame', slice: 'Slice',
    apexStart: 'Apical 起始 slice', setApex: '把目前 slice 設為 Apex', confirmApex: '確認建議值',
    sliceOrder: 'Slice 順序', baseFirst: 'base → apex', apexFirst: 'apex → base',
    results: '分割檢查', noResults: '分割完成後會在這裡列出需要看一下的 slice。',
    st_pending: '待處理', st_queued: '排隊中', st_running: '分割中', st_review: '待確認 Apex',
    st_done: '完成', st_error: '錯誤',
    suggested: '自動建議', confirmed: '已確認',
    lv_basal: 'basal', lv_mid: 'mid', lv_apical: 'apical', lv_apex: 'apex cap',
    added: n => `加入 ${n} 個病例`, queued: n => `${n} 個病例排入佇列`,
    downloading: (m, p) => `下載模型 ${m} … ${p}`, runningCase: (n, p) => `分割中：${n} ${p}`,
    queueLeft: n => `，佇列剩 ${n}`, idle: '待命', apexSaved: k => `Apex 設為 slice ${k}，AHA 已重算`,
    confirmRemove: n => `從清單移除 ${n} 個病例？（不會刪除檔案）`,
    notSegmented: '尚未分割。', error: '錯誤',
    lblLV: 'LV 血池', lblMYO: '心肌', lblRV: 'RV 血池', basal: 'Basal', mid: 'Mid', apical: 'Apical', apex: 'Apex',
    modelMissing: '（首次使用時下載）',
    phases: (ed, es) => `ED frame ${ed} · ES frame ${es}（LV 血池最大 / 最小）`,
    checksTitle: '建議檢查', show: '查看', flagged: '有警示',
    checksNone: '自動檢查沒有發現問題。仍建議切到「輪廓」逐張看一下。',
    checksNote: '自動檢查只抓得到明顯的錯誤：ES 比 ED 大、漏標的 slice、基底 LV 沒被心肌包住。',
    singleFrame: '單一時相：需要 cine 才能做 ED/ES 的比對檢查。',
    chkBigger: (z, ch, es, ed) => `slice ${z}：ES 的 ${ch} 比 ED 大（${es} vs ${ed} mm²），可能標到心室外的結構`,
    chkGap: (z, ph, ch, a, n) => `slice ${z}：${ph} 的 ${ch} 疑似漏標（${a} mm²，鄰近 slice ${n} mm²）`,
    chkRing: (z, p) => `slice ${z}：ES 最基底的 LV 只有 ${p}% 被心肌包住，LV 標記可能已進入心房`,
    outputs: '輸出檔案',
  },
  en: {
    addFiles: '＋ Files', addFolder: '＋ Folder', model: 'Model', stop: '■ Stop',
    runAll: n => `▶ Segment pending (${n})`, runChecked: n => `▶ Segment checked (${n})`,
    settings: 'Settings', pathsLabel: 'File or folder paths (one per line)',
    include: 'Include keywords (comma separated, empty = all)', exclude: 'Exclude keywords', add: 'Add', cancel: 'Cancel',
    saveMat: 'Write legacy .mat to the workspace', matInput: 'Also write .mat next to the input',
    niiInput: 'Write segmentation .nii.gz next to the input', workspace: 'Workspace', openFolder: 'Open folder',
    cases: 'Cases', colName: 'Name', colStatus: 'Status', colApex: 'Apex', colChecks: 'QC',
    emptyCases: 'Use “＋ Files” or “＋ Folder” to add cine NIfTI images.\nNo CSV and no apex needed up front: confirm the apex on the images after segmentation.',
    remove: 'Remove checked', openCaseFolder: 'Open case folder',
    viewSingle: 'Single', viewMosaic: 'All slices', zoomHeart: 'Zoom to heart', zoomFull: 'Full image',
    ovSeg: 'Segmentation', ovEdge: 'Contours', ovAha: 'AHA 17', ovNone: 'No overlay',
    opacity: 'Opacity', hintSelect: 'Select a case on the left', frame: 'Frame', slice: 'Slice',
    apexStart: 'First apical slice', setApex: 'Use current slice as apex', confirmApex: 'Confirm suggestion',
    sliceOrder: 'Slice order', baseFirst: 'base → apex', apexFirst: 'apex → base',
    results: 'Segmentation check', noResults: 'Slices worth a look are listed here after segmentation.',
    st_pending: 'Pending', st_queued: 'Queued', st_running: 'Segmenting', st_review: 'Check apex',
    st_done: 'Done', st_error: 'Error',
    suggested: 'suggested', confirmed: 'confirmed',
    lv_basal: 'basal', lv_mid: 'mid', lv_apical: 'apical', lv_apex: 'apex cap',
    added: n => `Added ${n} case(s)`, queued: n => `Queued ${n} case(s)`,
    downloading: (m, p) => `Downloading model ${m} … ${p}`, runningCase: (n, p) => `Segmenting: ${n} ${p}`,
    queueLeft: n => `, ${n} queued`, idle: 'Idle', apexSaved: k => `Apex set to slice ${k}; AHA updated`,
    confirmRemove: n => `Remove ${n} case(s) from the list? (files are kept)`,
    notSegmented: 'Not segmented yet.', error: 'Error',
    lblLV: 'LV blood', lblMYO: 'Myocardium', lblRV: 'RV blood', basal: 'Basal', mid: 'Mid', apical: 'Apical', apex: 'Apex',
    modelMissing: ' (downloaded on first use)',
    phases: (ed, es) => `ED frame ${ed} · ES frame ${es} (largest / smallest LV blood pool)`,
    checksTitle: 'Worth checking', show: 'Show', flagged: 'flagged',
    checksNone: 'The automatic checks found nothing. Still worth stepping through the slices with Contours.',
    checksNote: 'The automatic checks only catch clear errors: a cavity larger at ES than at ED, a missed slice, a basal LV not enclosed by myocardium.',
    singleFrame: 'Single frame: the ED/ES comparisons need a cine.',
    chkBigger: (z, ch, es, ed) => `slice ${z}: ${ch} larger at ES than at ED (${es} vs ${ed} mm²), the label may cover a structure outside the ventricle`,
    chkGap: (z, ph, ch, a, n) => `slice ${z}: ${ch} possibly missed at ${ph} (${a} mm², neighbouring slices ${n} mm²)`,
    chkRing: (z, p) => `slice ${z}: only ${p}% of the basal-most LV at ES is enclosed by myocardium, the LV label may include atrium`,
    outputs: 'Output files',
  },
};
let lang = 'en';
try { lang = localStorage.getItem('tigerhx_lang') || 'en'; } catch (e) { /* storage blocked */ }
if (!I18N[lang]) lang = 'en';
const t = (k, ...a) => { const v = I18N[lang][k]; return typeof v === 'function' ? v(...a) : (v ?? k); };

const AHA_NAMES = [
  '', 'basal anterior', 'basal anteroseptal', 'basal inferoseptal', 'basal inferior',
  'basal inferolateral', 'basal anterolateral', 'mid anterior', 'mid anteroseptal',
  'mid inferoseptal', 'mid inferior', 'mid inferolateral', 'mid anterolateral',
  'apical anterior', 'apical septal', 'apical inferior', 'apical lateral', 'apex',
];

// Overlay colours: drawn on the black viewer, so the dark-surface steps.
const SEG_COLORS = [[0, 0, 0], [57, 135, 229], [217, 89, 38], [25, 158, 112]];
// AHA: hue encodes the level, alternating lightness separates neighbouring sectors.
const AHA_LEVEL = [
  [[183, 211, 246], [28, 92, 171]],    // basal: blue
  [[250, 196, 170], [184, 70, 25]],    // mid: orange
  [[160, 230, 203], [16, 120, 84]],    // apical: aqua
];
const AHA_APEX = [144, 133, 233];      // 17: violet

function lut(colors, names) {
  const n = colors.length;
  return {
    R: colors.map(c => c[0]), G: colors.map(c => c[1]), B: colors.map(c => c[2]),
    A: colors.map((_, i) => (i === 0 ? 0 : 255)), I: [...Array(n).keys()], labels: names,
  };
}
function ahaColors() {
  const out = [[0, 0, 0]];
  for (let s = 1; s <= 16; s++) {
    const level = s <= 6 ? 0 : s <= 12 ? 1 : 2;
    out.push(AHA_LEVEL[level][s % 2]);
  }
  out.push(AHA_APEX);
  return out;
}

// ---------------------------------------------------------------- state
const S = {
  state: null, selected: null, checked: new Set(),
  view: 'single', zoom: 'heart', overlay: 'seg', opacity: 0.45,
  frame: 0, slice: null, playing: null,
  loadedKey: null, ahaKey: null, loading: false, message: '', error: false,
};
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function say(msg, isError = false) {
  S.message = msg; S.error = isError;
  const el = $('statusMsg');
  el.textContent = msg;
  el.className = isError ? 'err' : '';
}
const guard = fn => async (...a) => { try { await fn(...a); } catch (e) { say(`${t('error')}: ${e.message}`, true); } };

const selectedCase = () => S.state && S.state.cases.find(c => c.id === S.selected);
const isSegmented = c => !!(c && ['review', 'done'].includes(c.status) && c.levels);

// ---------------------------------------------------------------- polling
async function refresh() {
  S.state = await api('/api/state');
  render();
}
async function poll() {
  try { await refresh(); } catch (e) { say(`${t('error')}: ${e.message}`, true); }
  const busy = S.state && (S.state.running || S.state.queued || S.state.model_status.state === 'downloading');
  setTimeout(poll, busy ? 500 : 1500);
}

// ---------------------------------------------------------------- render
function applyI18n() {
  document.documentElement.lang = lang === 'zh' ? 'zh-Hant' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  $('btnLang').textContent = lang === 'zh' ? 'EN' : '中文';
}

function render() {
  const st = S.state;
  if (!st) return;
  if (!selectedCase() && st.cases.length) S.selected = st.cases[0].id;
  renderTop(st);
  renderCases(st);
  renderStatus(st);
  const c = selectedCase();
  renderViewerBars(c);
  renderResults(c);
  if (c) maybeLoadCase(c);
}

function renderTop(st) {
  const sel = $('model');
  const want = st.models.map(m => m.name + (m.downloaded ? '' : t('modelMissing')));
  if (sel.options.length !== want.length || [...sel.options].some((o, i) => o.textContent !== want[i])) {
    sel.innerHTML = '';
    st.models.forEach((m, i) => {
      const o = document.createElement('option');
      o.value = m.name; o.textContent = want[i];
      sel.appendChild(o);
    });
  }
  if (document.activeElement !== sel) sel.value = st.settings.model;
  document.querySelectorAll('[data-setting]').forEach(el => { el.checked = !!st.settings[el.dataset.setting]; });
  $('workspace').textContent = st.workspace;

  const checked = [...S.checked].filter(id => st.cases.some(c => c.id === id));
  const pending = st.cases.filter(c => c.status === 'pending' || c.status === 'error').length;
  const run = $('btnRun');
  run.textContent = checked.length ? t('runChecked', checked.length) : t('runAll', pending);
  run.disabled = checked.length ? false : pending === 0;
  $('btnStop').disabled = !(st.running || st.queued);
}

function chip(c) {
  const pct = c.status === 'running' ? ` ${Math.round(c.progress * 100)}%` : '';
  return `<span class="chip ${c.status}"><span class="dot"></span>${esc(t('st_' + c.status))}${pct}</span>` +
    (c.status === 'running' ? `<div class="bar"><i style="width:${(c.progress * 100).toFixed(1)}%"></i></div>` : '');
}

function renderCases(st) {
  const body = $('caseBody');
  $('caseCount').textContent = st.cases.length ? `(${st.cases.length})` : '';
  $('caseEmpty').classList.toggle('hidden', st.cases.length > 0);
  $('caseEmpty').innerText = t('emptyCases');
  const rows = st.cases.map(c => {
    const apex = c.apex_slice === null ? '–' : (c.apex_confirmed ? `${c.apex_slice}` : `${c.apex_slice}?`);
    const nWarn = (c.checks || []).length;
    const qc = !isSegmented(c) ? '–' : nWarn
      ? `<span class="warn-mark" title="${esc(t('checksTitle'))}: ${nWarn}">⚠ ${nWarn}</span>` : '<span class="ok-mark">✓</span>';
    const msg = c.message && c.status === 'error' ? `<div class="case-msg">${esc(c.message)}</div>` : '';
    return `<tr class="row${c.id === S.selected ? ' selected' : ''}" data-id="${c.id}" title="${esc(c.path)}">
      <td><input type="checkbox" data-check="${c.id}"${S.checked.has(c.id) ? ' checked' : ''}></td>
      <td class="name">${esc(c.name)}${msg}</td><td>${chip(c)}</td>
      <td class="num">${esc(apex)}</td><td class="num">${qc}</td></tr>`;
  }).join('');
  if (body.dataset.html !== rows) { body.innerHTML = rows; body.dataset.html = rows; }
  $('checkAll').checked = st.cases.length > 0 && st.cases.every(c => S.checked.has(c.id));
  $('btnRemove').disabled = S.checked.size === 0;
  $('btnOpenCase').disabled = !isSegmented(selectedCase());
}

function renderStatus(st) {
  const ms = st.model_status;
  let main = t('idle');
  if (ms.state === 'downloading') main = t('downloading', ms.name, `${Math.round(ms.progress * 100)}%`);
  else if (st.running) {
    const c = st.cases.find(x => x.id === st.running);
    if (c) main = t('runningCase', c.name, `${Math.round(c.progress * 100)}%`);
  }
  if (st.queued) main += t('queueLeft', st.queued);
  if (ms.state === 'error') main += ` · ${ms.message}`;
  $('statusMain').textContent = main;
}

function renderViewerBars(c) {
  const seg = isSegmented(c);
  const nt = c && c.shape ? c.shape[3] : 1;
  const nz = c && c.shape ? c.shape[2] : 0;
  const fr = $('frame');
  fr.max = Math.max(0, nt - 1);
  if (+fr.value !== S.frame) fr.value = S.frame;
  $('frameText').textContent = c && c.shape ? `${S.frame} / ${nt - 1}` : '–';
  $('btnED').disabled = !seg || nt < 2;
  $('btnES').disabled = !seg || nt < 2;
  $('btnPlay').disabled = !S.loadedKey || nt < 2;
  $('sliceText').textContent = S.slice === null || !nz ? '–' : `${S.slice} / ${nz - 1}`;
  $('btnSlicePrev').disabled = !S.loadedKey || S.view === 'mosaic';
  $('btnSliceNext').disabled = !S.loadedKey || S.view === 'mosaic';
  fr.disabled = !S.loadedKey || nt < 2;
  const level = seg && S.slice !== null && c.levels ? c.levels[S.slice] : null;
  const notes = level ? [t('lv_' + level)] : [];
  const flagged = seg && (c.checks || []).some(k => k.slice === S.slice && frameOf(c, k) === S.frame);
  if (flagged) notes.push(`⚠ ${t('flagged')}`);
  $('levelTag').classList.toggle('hidden', !notes.length);
  $('levelTag').classList.toggle('warn', flagged);
  $('levelTag').textContent = notes.join(' · ');

  $('apexValue').textContent = seg && c.apex_slice !== null ? c.apex_slice : '–';
  const tag = $('apexTag');
  tag.classList.toggle('hidden', !seg || c.apex_slice === null);
  tag.textContent = seg && c.apex_confirmed ? t('confirmed') : t('suggested');
  tag.className = 'tag' + (seg && c.apex_confirmed ? ' ok' : ' warn') + (tag.classList.contains('hidden') ? ' hidden' : '');
  $('btnSetApex').disabled = !seg || S.slice === null;
  $('btnConfirmApex').disabled = !seg || c.apex_confirmed || c.apex_slice === null;
  const so = $('sliceOrder');
  so.disabled = !seg;
  if (seg && document.activeElement !== so) so.value = c.base_first === false ? '0' : '1';
  document.querySelectorAll('#viewMode button').forEach(b => b.classList.toggle('on', b.dataset.v === S.view));
  document.querySelectorAll('#zoomMode button').forEach(b => {
    b.classList.toggle('on', b.dataset.v === (seg && c.crop ? S.zoom : 'full'));
    b.disabled = !seg || !c.crop;
  });
  document.querySelectorAll('#overlayMode button').forEach(b => {
    b.classList.toggle('on', b.dataset.v === S.overlay);
    b.disabled = b.dataset.v !== 'none' && !seg;
  });
}

// ---------------------------------------------------------------- results
const frameOf = (c, k) => (k.phase === 'ED' ? c.ed : c.es);

function checkText(k) {
  if (k.kind === 'es_gt_ed') return t('chkBigger', k.slice, k.chamber, k.es, k.ed);
  if (k.kind === 'gap') return t('chkGap', k.slice, k.phase, k.chamber, k.area, k.neighbours);
  if (k.kind === 'open_ring') return t('chkRing', k.slice, Math.round(k.coverage * 100));
  return `slice ${k.slice}: ${k.kind}`;
}

function renderResults(c) {
  const box = $('results');
  const key = c ? JSON.stringify([c.id, c.status, c.model, c.ed, c.es, c.checks, c.outputs, lang, c.message]) : 'none';
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  if (!c) { box.innerHTML = `<div class="empty">${esc(t('noResults'))}</div>`; return; }
  const info = `<div class="info"><b>${esc(c.name)}</b><br><span class="mono">${esc(c.path)}</span></div>`;
  if (!isSegmented(c)) {
    const msg = c.status === 'error' ? `<div class="case-msg">${esc(c.message)}</div>` : '';
    box.innerHTML = `${info}<div class="empty">${esc(t('notSegmented'))}</div>${msg}`;
    return;
  }
  const multi = c.shape[3] > 1;
  const checks = (c.checks || []).map(k => `<li class="warn"><span class="ico">⚠</span>
      <span>${esc(checkText(k))}</span>
      <button class="small" data-goto="${frameOf(c, k)},${k.slice}">${esc(t('show'))}</button></li>`).join('');
  const outputs = Object.values(c.outputs || {}).map(p => `<li class="mono">${esc(p)}</li>`).join('');
  box.innerHTML = `${info}
    <div class="info">${esc(c.model)}${multi ? `<br>${esc(t('phases', c.ed, c.es))}` : ''}</div>
    <div class="card">
      <div class="card-title">${esc(t('checksTitle'))}</div>
      ${!multi ? `<div class="info">${esc(t('singleFrame'))}</div>`
        : checks ? `<ul class="checks">${checks}</ul>`
        : `<ul class="checks"><li class="done"><span class="ico">✓</span><span>${esc(t('checksNone'))}</span></li></ul>`}
      <div class="info">${esc(t('checksNote'))}</div>
    </div>
    ${outputs ? `<div class="sub-title">${esc(t('outputs'))}</div><ul class="outputs">${outputs}</ul>` : ''}`;
}

async function gotoFrameSlice(f, k) {
  if (S.view === 'mosaic') { S.view = 'single'; applyView(); }
  setFrame(f);
  if (k !== '' && k !== undefined && k !== null) goToSlice(+k);
}

// ---------------------------------------------------------------- viewer
const nv = new niivue.Niivue({
  backColor: [0, 0, 0, 1], show3Dcrosshair: false, isColorbar: false, isOrientCube: false,
  isOrientationTextVisible: false, dragAndDropEnabled: false, loadingText: '',
  isRadiologicalConvention: false, crosshairWidth: 0.25, crosshairGap: 8,
});
nv.attachToCanvas($('gl'));
nv.setCrosshairColor([1, 0.85, 0.2, 0.6]);
nv.onLocationChange = d => {
  if (!d || !d.vox) return;
  const k = Math.round(d.vox[2]);
  if (k !== S.slice) { S.slice = k; renderViewerBars(selectedCase()); }
  showReadout(d);
};

function showReadout(d) {
  const el = $('readout');
  const c = selectedCase();
  if (!c || !d.values) { el.classList.add('hidden'); return; }
  const val = name => { const v = d.values.find(x => x.name === name); return v ? Math.round(v.value) : 0; };
  const parts = [`slice ${S.slice}`];
  if (S.overlay === 'seg' || S.overlay === 'edge') {
    const s = val('seg.nii');
    if (s) parts.push([null, t('lblLV'), t('lblMYO'), t('lblRV')][s]);
  } else if (S.overlay === 'aha') {
    const a = val('aha.nii');
    if (a) parts.push(`AHA ${a} · ${AHA_NAMES[a]}`);
  }
  el.textContent = parts.join(' · ');
  el.classList.remove('hidden');
}

function renderLegend() {
  const el = $('legend');
  if (!S.loadedKey || S.overlay === 'none' || !volIndex('seg.nii')) { el.classList.add('hidden'); return; }
  const items = S.overlay === 'seg' || S.overlay === 'edge'
    ? [[SEG_COLORS[1], t('lblLV')], [SEG_COLORS[2], t('lblMYO')], [SEG_COLORS[3], t('lblRV')]]
    : [[AHA_LEVEL[0][1], `${t('basal')} 1–6`], [AHA_LEVEL[1][1], `${t('mid')} 7–12`],
      [AHA_LEVEL[2][1], `${t('apical')} 13–16`], [AHA_APEX, `${t('apex')} 17`]];
  el.replaceChildren(...items.map(([c, label]) => {
    const s = document.createElement('span');
    const i = document.createElement('i'); i.style.background = `rgb(${c.join(',')})`;
    s.append(i, document.createTextNode(label));
    return s;
  }));
  el.classList.remove('hidden');
}

const zoomed = c => isSegmented(c) && !!c.crop && S.zoom === 'heart';
const volUrl = (c, kind, v = '') => {
  const q = new URLSearchParams();
  if (zoomed(c)) q.set('crop', '1');
  if (v) q.set('v', v);
  return `/api/cases/${c.id}/volume/${kind}.nii${q.toString() ? `?${q}` : ''}`;
};
const ahaVersion = c => `${c.apex_slice}-${c.base_first}-${c.model}`;
const volIndex = name => { const i = nv.volumes.findIndex(v => v.name === name); return i < 0 ? 0 : i; };
const segLut = () => lut(SEG_COLORS, ['', 'LV', 'MYO', 'RV']);
const ahaLut = () => lut(ahaColors(), AHA_NAMES.map((n, i) => (i ? `${i}` : '')));

async function maybeLoadCase(c) {
  const seg = isSegmented(c);
  const key = `${c.id}|${seg ? c.model : 'raw'}|${zoomed(c) ? c.crop.join(',') : 'full'}`;
  if (S.loading) return;
  if (key === S.loadedKey) {
    if (seg && ahaVersion(c) !== S.ahaKey) await reloadAha(c);
    return;
  }
  // a zoom change keeps the frame and slice; a new case or segmentation starts over
  const samePlace = S.loadedKey && S.loadedKey.split('|').slice(0, 2).join('|') === key.split('|').slice(0, 2).join('|');
  S.loading = true;
  try {
    $('canvasHint').textContent = '…';
    $('canvasHint').classList.remove('hidden');
    stopPlay();
    const stamp = Date.now();
    const vols = [{ url: volUrl(c, 'image'), name: 'image.nii' }];
    if (seg) {
      vols.push({ url: volUrl(c, 'seg', stamp), name: 'seg.nii' });
      vols.push({ url: volUrl(c, 'edge', stamp), name: 'edge.nii' });
      vols.push({ url: volUrl(c, 'aha', ahaVersion(c)), name: 'aha.nii' });
    }
    await nv.loadVolumes(vols);
    if (seg) {
      nv.volumes[volIndex('seg.nii')].setColormapLabel(segLut());
      nv.volumes[volIndex('edge.nii')].setColormapLabel(segLut());
      nv.volumes[volIndex('aha.nii')].setColormapLabel(ahaLut());
      S.ahaKey = ahaVersion(c);
    } else if (S.overlay !== 'none') {
      S.overlay = 'seg';
    }
    S.loadedKey = key;
    const nz = c.shape ? c.shape[2] : nv.volumes[0].hdr.dims[3];
    if (!samePlace) {
      S.frame = seg && c.ed !== null ? c.ed : 0;
      S.slice = seg && c.levels ? c.levels.findIndex(l => l === 'mid') : Math.floor(nz / 2);
      if (S.slice < 0) S.slice = Math.floor(nz / 2);
    }
    applyOverlay();
    applyFrame();
    applyView();
    goToSlice(S.slice);
    $('canvasHint').classList.add('hidden');
  } catch (e) {
    S.loadedKey = null;
    $('canvasHint').textContent = `${t('error')}: ${e.message}`;
  } finally {
    S.loading = false;
  }
  render();
}

async function reloadAha(c) {
  const old = nv.volumes.find(v => v.name === 'aha.nii');
  if (!old) return;
  S.loading = true;
  try {
    const vol = await niivue.NVImage.loadFromUrl({ url: volUrl(c, 'aha', ahaVersion(c)), name: 'aha.nii' });
    vol.setColormapLabel(ahaLut());
    nv.removeVolume(old);
    nv.addVolume(vol);
    S.ahaKey = ahaVersion(c);
    applyOverlay();
    applyFrame();
  } finally {
    S.loading = false;
  }
}

function applyOverlay() {
  [['seg.nii', 'seg', S.opacity], ['edge.nii', 'edge', 1], ['aha.nii', 'aha', S.opacity]].forEach(([name, mode, op]) => {
    if (nv.volumes.some(v => v.name === name)) nv.setOpacity(volIndex(name), S.overlay === mode ? op : 0);
  });
  renderLegend();
}
function applyFrame() {
  nv.volumes.forEach(v => { if (v.nFrame4D > 1) nv.setFrame4D(v.id, S.frame); });
}
function applyView() {
  const c = selectedCase();
  if (S.view === 'mosaic' && nv.volumes.length) {
    const [, nx, ny, nz] = nv.volumes[0].hdr.dims;  // cropped when zoomed
    const [dx, dy, dz] = c.zooms || [1, 1, 1];
    const canvas = $('gl');
    const W = canvas.clientWidth, H = Math.max(1, canvas.clientHeight);
    const tw = ny * dy, th = nx * dx;  // tile width / height in mm
    let cols = 1, best = 0;
    for (let c = 1; c <= nz; c++) {  // the column count giving the largest tiles
      const scale = Math.min(W / (c * tw), H / (Math.ceil(nz / c) * th));
      if (scale > best) { best = scale; cols = c; }
    }
    const rows = [];
    for (let k = 0; k < nz; k += cols) {
      rows.push([...Array(Math.min(cols, nz - k)).keys()].map(i => ((k + i) * dz).toFixed(3)).join(' '));
    }
    nv.setSliceMosaicString('A ' + rows.join(' ; '));
  } else {
    nv.setSliceMosaicString('');
    nv.setSliceType(nv.sliceTypeAxial);
  }
}
function goToSlice(k) {
  const c = selectedCase();
  if (!c || !nv.volumes.length) return;
  const nz = c.shape ? c.shape[2] : nv.volumes[0].hdr.dims[3];
  k = Math.max(0, Math.min(nz - 1, k));
  const pos = nv.scene.crosshairPos;
  nv.scene.crosshairPos = [pos[0], pos[1], (k + 0.5) / nz];
  nv.drawScene();
  nv.createOnLocationChange();
  S.slice = k;
  renderViewerBars(c);
}
function setFrame(f) {
  const c = selectedCase();
  if (!c || !c.shape) return;
  S.frame = Math.max(0, Math.min(c.shape[3] - 1, f));
  applyFrame();
  renderViewerBars(c);
}
function stopPlay() {
  if (S.playing) { clearInterval(S.playing); S.playing = null; $('btnPlay').textContent = '▶'; }
}
function togglePlay() {
  const c = selectedCase();
  if (S.playing) return stopPlay();
  if (!c || !c.shape || c.shape[3] < 2) return;
  $('btnPlay').textContent = '❚❚';
  S.playing = setInterval(() => setFrame((S.frame + 1) % c.shape[3]), 90);
}

// ---------------------------------------------------------------- actions
async function addPaths(paths, include = '', exclude = '') {
  const r = await api('/api/cases/add', { paths, include, exclude });
  say(t('added', r.added));
  await refresh();
}
const hasNative = () => window.pywebview && window.pywebview.api;

function showAddPanel(paths = []) {
  $('addPaths').value = paths.join('\n');
  $('addPanel').classList.remove('hidden');
  $('addPaths').focus();
}

function bind() {
  $('btnAddFiles').onclick = guard(async () => {
    if (!hasNative()) return showAddPanel();
    const files = await window.pywebview.api.pick_files();
    if (files && files.length) await addPaths(files);
  });
  $('btnAddFolder').onclick = guard(async () => {
    if (!hasNative()) return showAddPanel();
    const dirs = await window.pywebview.api.pick_folder();
    if (dirs && dirs.length) showAddPanel(dirs);
  });
  $('btnAddConfirm').onclick = guard(async () => {
    const paths = $('addPaths').value.split('\n').map(s => s.trim()).filter(Boolean);
    if (!paths.length) return;
    await addPaths(paths, $('addInclude').value, $('addExclude').value);
    $('addPanel').classList.add('hidden');
  });
  $('btnAddCancel').onclick = () => $('addPanel').classList.add('hidden');
  $('btnSettings').onclick = () => $('settingsPanel').classList.toggle('hidden');
  document.querySelectorAll('[data-setting]').forEach(el => {
    el.onchange = guard(async () => { await api('/api/settings', { [el.dataset.setting]: el.checked }); await refresh(); });
  });
  $('model').onchange = guard(async () => { await api('/api/settings', { model: $('model').value }); await refresh(); });
  $('btnRun').onclick = guard(async () => {
    const ids = [...S.checked].filter(id => S.state.cases.some(c => c.id === id));
    const r = await api('/api/run', { ids });
    say(t('queued', r.queued));
    S.checked.clear();
    await refresh();
  });
  $('btnStop').onclick = guard(async () => { await api('/api/stop', {}); await refresh(); });
  $('btnOpenWs').onclick = guard(() => api('/api/open', {}));
  $('btnOpenCase').onclick = guard(() => api('/api/open', { id: S.selected }));
  $('btnRemove').onclick = guard(async () => {
    const ids = [...S.checked];
    if (!ids.length || !window.confirm(t('confirmRemove', ids.length))) return;
    await api('/api/cases/remove', { ids });
    S.checked.clear();
    if (ids.includes(S.selected)) { S.selected = null; S.loadedKey = null; nv.volumes.slice().forEach(v => nv.removeVolume(v)); }
    await refresh();
  });
  $('btnLang').onclick = () => {
    lang = lang === 'zh' ? 'en' : 'zh';
    try { localStorage.setItem('tigerhx_lang', lang); } catch (e) { /* ignore */ }
    applyI18n();
    $('caseBody').dataset.html = '';
    render();
    renderLegend();
  };

  $('caseBody').addEventListener('click', ev => {
    const cb = ev.target.closest('[data-check]');
    if (cb) {
      if (cb.checked) S.checked.add(cb.dataset.check); else S.checked.delete(cb.dataset.check);
      $('caseBody').dataset.html = '';
      return render();
    }
    const row = ev.target.closest('tr.row');
    if (row && row.dataset.id !== S.selected) {
      S.selected = row.dataset.id;
      $('caseBody').dataset.html = '';
      render();
    }
  });
  $('checkAll').onchange = () => {
    S.checked = $('checkAll').checked ? new Set(S.state.cases.map(c => c.id)) : new Set();
    $('caseBody').dataset.html = '';
    render();
  };

  document.querySelectorAll('#viewMode button').forEach(b => {
    b.onclick = () => { S.view = b.dataset.v; applyView(); renderViewerBars(selectedCase()); };
  });
  document.querySelectorAll('#zoomMode button').forEach(b => {
    b.onclick = () => { S.zoom = b.dataset.v; renderViewerBars(selectedCase()); render(); };
  });
  document.querySelectorAll('#overlayMode button').forEach(b => {
    b.onclick = () => { S.overlay = b.dataset.v; applyOverlay(); renderViewerBars(selectedCase()); };
  });
  $('opacity').oninput = () => { S.opacity = +$('opacity').value; applyOverlay(); };
  $('frame').oninput = () => setFrame(+$('frame').value);
  $('btnPlay').onclick = togglePlay;
  $('btnED').onclick = () => { const c = selectedCase(); if (c) setFrame(c.ed); };
  $('btnES').onclick = () => { const c = selectedCase(); if (c) setFrame(c.es); };
  $('btnSlicePrev').onclick = () => goToSlice((S.slice ?? 0) - 1);
  $('btnSliceNext').onclick = () => goToSlice((S.slice ?? 0) + 1);

  const postApex = async (apex, baseFirst) => {
    const c = selectedCase();
    await api(`/api/cases/${c.id}/apex`, { apex_slice: apex, base_first: baseFirst });
    say(t('apexSaved', apex));
    await refresh();
  };
  $('btnSetApex').onclick = guard(() => postApex(S.slice, null));
  $('btnConfirmApex').onclick = guard(() => postApex(selectedCase().apex_slice, null));
  $('results').addEventListener('click', ev => {
    const b = ev.target.closest('button[data-goto]');
    if (!b || !selectedCase()) return;
    const [f, k] = b.dataset.goto.split(',');
    gotoFrameSlice(+f, k);
  });
  $('sliceOrder').onchange = guard(() => postApex(selectedCase().apex_slice, $('sliceOrder').value === '1'));

  document.addEventListener('keydown', ev => {
    if (ev.target.closest('input, textarea, select') || !S.loadedKey) return;
    const keys = {
      ArrowLeft: () => setFrame(S.frame - 1), ArrowRight: () => setFrame(S.frame + 1),
      ArrowUp: () => goToSlice(S.slice - 1), ArrowDown: () => goToSlice(S.slice + 1), ' ': togglePlay,
    };
    if (keys[ev.key]) { ev.preventDefault(); keys[ev.key](); }
  });
  window.addEventListener('resize', () => { if (S.view === 'mosaic') applyView(); });
}

applyI18n();
bind();
poll();
