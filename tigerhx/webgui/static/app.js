'use strict';

// ---------------------------------------------------------------- strings
const I18N = {
  zh: {
    addFiles: '＋ 檔案', addFolder: '＋ 資料夾', model: '模型', stop: '■ 停止',
    runAll: n => `▶ 分割待處理 (${n})`, runChecked: n => `▶ 分割已選 (${n})`,
    settings: '設定', pathsLabel: '檔案或資料夾路徑（每行一個）',
    include: '包含關鍵字（逗號分隔，空白=全部）', exclude: '排除關鍵字', add: '加入', cancel: '取消',
    saveMat: '輸出 .mat 到輸出資料夾（舊版格式）', matInput: '.mat 也存到原始資料夾',
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
    lblLV: 'LV 血池', lblMYO: '心肌', lblRV: 'RV 血池', lblEndo: 'LV 心內膜', lblEpi: 'LV 心外膜',
    ovSectors: '6/4 分區', ovLV: 'LV', ovRV: 'RV', overlay: '疊圖', cmap: '色表',
    openResults: '開啟結果 (.mat)', genCsv: '產生 CSV (GenCSV)', resultsPrompt: '含 *pred*.mat 的資料夾',
    addedFull: (a, u, m) => `加入 ${a} 個病例` + (u ? `，更新 ${u} 個` : '') + (m ? `，${m} 個檔案找不到（見 Log）` : ''),
    genCsvDone: (n, p) => `GenCSV：${n} 個檔案 → ${p}（填好 Apex 後用「＋ 檔案」加入這個 CSV）`,
    outputDir: '.mat 輸出資料夾', outputDirHint: '空白 = 工作資料夾/output', csvDir: 'CSV 資料夾',
    downloadAll: '下載全部模型', downloadStarted: n => (n ? `開始下載 ${n} 個模型` : '全部模型都已下載'),
    log: 'Log', kindMat: '舊結果', matNote: '舊版 GUI 的結果（唯讀）：apex 取自其 6/4 分區；不會寫出任何檔案。',
    csvNote: '來自 CSV：', apexFromCsv: 'apex 已確認', ownOutputs: (m, n) => `.mat 存原始資料夾 ${m}、.nii.gz 存原始資料夾 ${n}`,
    yes: '是', no: '否', sectorsLegend: (n) => `${n} 分區`,
    wall_anterior: '前壁', wall_anteroseptal: '前間隔', wall_inferoseptal: '下間隔', wall_inferior: '下壁',
    wall_inferolateral: '下側壁', wall_anterolateral: '前側壁', wall_septal: '間隔', wall_lateral: '側壁', wall_apex: '心尖',
    modelMissing: '（首次使用時下載）',
    phases: (ed, es) => `ED frame ${ed} · ES frame ${es}（LV 血池最大 / 最小）`,
    checksTitle: '建議檢查', show: '查看', flagged: '有警示',
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
    saveMat: 'Write legacy .mat to the output folder', matInput: 'Also write .mat next to the input',
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
    lblLV: 'LV blood', lblMYO: 'Myocardium', lblRV: 'RV blood', lblEndo: 'LV endocardium', lblEpi: 'LV epicardium',
    ovSectors: '6/4 sectors', ovLV: 'LV', ovRV: 'RV', overlay: 'Overlay', cmap: 'Colour map',
    openResults: 'Open results (.mat)', genCsv: 'Generate CSV (GenCSV)', resultsPrompt: 'Folder with *pred*.mat files',
    addedFull: (a, u, m) => `Added ${a} case(s)` + (u ? `, updated ${u}` : '') + (m ? `, ${m} file(s) not found (see Log)` : ''),
    genCsvDone: (n, p) => `GenCSV: ${n} file(s) → ${p} (fill in Apex, then add the CSV with “＋ Files”)`,
    outputDir: '.mat output folder', outputDirHint: 'empty = workspace/output', csvDir: 'CSV folder',
    downloadAll: 'Download all models', downloadStarted: n => (n ? `Downloading ${n} model(s)` : 'All models are already downloaded'),
    log: 'Log', kindMat: 'result', matNote: 'Result of the original GUI (read-only): apex taken from its 6/4 sectors; nothing is written.',
    csvNote: 'From the CSV: ', apexFromCsv: 'apex confirmed', ownOutputs: (m, n) => `.mat next to input ${m}, .nii.gz next to input ${n}`,
    yes: 'yes', no: 'no', sectorsLegend: (n) => `${n} sectors`,
    wall_anterior: 'Anterior', wall_anteroseptal: 'Anteroseptal', wall_inferoseptal: 'Inferoseptal', wall_inferior: 'Inferior',
    wall_inferolateral: 'Inferolateral', wall_anterolateral: 'Anterolateral', wall_septal: 'Septal', wall_lateral: 'Lateral', wall_apex: 'Apex',
    modelMissing: ' (downloaded on first use)',
    phases: (ed, es) => `ED frame ${ed} · ES frame ${es} (largest / smallest LV blood pool)`,
    checksTitle: 'Worth checking', show: 'Show', flagged: 'flagged',
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

// Overlay colours on the black viewer: fills use the dark-surface steps, lines
// brighter versions of the same hues so a thin line stays visible.
const SEG_FILL = { LV: [57, 135, 229], MYO: [217, 89, 38], RV: [25, 158, 112] };
const SEG_LINE = { LV: [92, 166, 255], MYO: [255, 138, 76], RV: [52, 211, 153] };
// AHA: the hue follows the wall, so a wall keeps its colour at every level and
// neighbouring segments always differ; equal lightness keeps them balanced.
const WALL_COLORS = {
  anterior: [247, 133, 125], anteroseptal: [211, 163, 41], inferoseptal: [110, 194, 114],
  inferior: [0, 195, 213], inferolateral: [127, 168, 255], anterolateral: [215, 138, 219],
  septal: [168, 181, 64], lateral: [178, 152, 249], apex: [215, 215, 215],
};
const AHA_WALL = [null, 'anterior', 'anteroseptal', 'inferoseptal', 'inferior', 'inferolateral', 'anterolateral',
  'anterior', 'anteroseptal', 'inferoseptal', 'inferior', 'inferolateral', 'anterolateral',
  'anterior', 'septal', 'inferior', 'lateral', 'apex'];
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// ---------------------------------------------------------------- state
const S = {
  state: null, selected: null, checked: new Set(),
  view: 'single', zoom: 'heart', overlay: 'seg', opacity: 0.45,
  frame: 0, slice: null, playing: null,
  loadedKey: null, ahaKey: null, loading: false, message: '', error: false,
  cmap: 'gray', logId: 0, logOpen: false,
};
try { S.cmap = localStorage.getItem('tigerhx_cmap') || 'gray'; } catch (e) { /* storage blocked */ }
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
async function pollLog() {
  const r = await api(`/api/log?after=${S.logId}`);
  if (!r.entries || !r.entries.length) return;
  const box = $('logBody');
  const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  for (const e of r.entries) {
    const row = document.createElement('div');
    row.className = `log-${e.level}`;
    row.textContent = `${e.time}  ${e.text}`;
    box.appendChild(row);
    S.logId = e.id;
  }
  while (box.childElementCount > 2000) box.firstElementChild.remove();
  if (atEnd) box.scrollTop = box.scrollHeight;
  const last = r.entries[r.entries.length - 1];
  if (last.level === 'error' && !S.logOpen) $('btnLog').classList.add('alert');
}
async function poll() {
  try { await refresh(); await pollLog(); } catch (e) { say(`${t('error')}: ${e.message}`, true); }
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
  if (document.activeElement !== $('outputDir')) $('outputDir').value = st.settings.output_dir || '';
  $('outputDir').placeholder = `${st.output_dir}  (${t('outputDirHint')})`;

  const checked = [...S.checked].filter(id => st.cases.some(c => c.id === id));
  const pending = st.cases.filter(c => (c.status === 'pending' || c.status === 'error') && c.kind !== 'mat').length;
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
      <td class="name">${esc(c.name)}${c.kind === 'mat' ? ` <span class="tag">${esc(t('kindMat'))}</span>` : ''}${msg}</td><td>${chip(c)}</td>
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
  const ov = $('overlayMode');
  ov.disabled = !seg;
  if (document.activeElement !== ov) ov.value = seg ? S.overlay : 'none';
  if (document.activeElement !== $('cmap')) $('cmap').value = S.cmap;
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
  const key = c ? JSON.stringify([c.id, c.status, c.model, c.ed, c.es, c.checks, c.outputs, lang, c.message,
    c.kind, c.mat_in_inputdir, c.nii_in_inputdir, c.apex_confirmed]) : 'none';
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  if (!c) { box.innerHTML = `<div class="empty">${esc(t('noResults'))}</div>`; return; }
  const yn = v => (v === null || v === undefined ? '–' : t(v ? 'yes' : 'no'));
  const own = c.mat_in_inputdir !== null || c.nii_in_inputdir !== null;
  const info = `<div class="info"><b>${esc(c.name)}</b><br><span class="mono">${esc(c.path)}</span></div>` +
    (c.kind === 'mat' ? `<div class="info">${esc(t('matNote'))}</div>` : '') +
    (own ? `<div class="info">${esc(t('csvNote') + t('ownOutputs', yn(c.mat_in_inputdir), yn(c.nii_in_inputdir)))}</div>` : '');
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
    ${checks ? `<div class="card">
      <div class="card-title">${esc(t('checksTitle'))}</div>
      <ul class="checks">${checks}</ul>
    </div>` : ''}
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
  if (['seg', 'edge', 'lv', 'rv'].includes(S.overlay)) {
    const s = val('seg.nii');
    if (s) parts.push([null, t('lblLV'), t('lblMYO'), t('lblRV')][s]);
  } else if (S.overlay === 'aha') {
    const a = val('aha.nii');
    if (a) parts.push(`AHA ${a} · ${AHA_NAMES[a]}`);
  } else if (S.overlay === 'sectors') {
    const a = val('sectors.nii');
    if (a) parts.push(`${t('ovSectors')} · ${a}`);
  }
  el.textContent = parts.join(' · ');
  el.classList.remove('hidden');
}

function renderLegend(items = []) {
  const el = $('legend');
  const key = JSON.stringify(items);
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  el.classList.toggle('hidden', !items.length);
  el.replaceChildren(...items.map(([c, label]) => {
    const s = document.createElement('span');
    const i = document.createElement('i');
    if (c) i.style.background = rgb(c); else i.style.visibility = 'hidden';
    s.append(i, document.createTextNode(label));
    return s;
  }));
}

const zoomed = c => isSegmented(c) && !!c.crop && S.zoom === 'heart';
const volUrl = (c, kind, v = '') => {
  const q = new URLSearchParams();
  if (zoomed(c)) q.set('crop', '1');
  if (v) q.set('v', v);
  return `/api/cases/${c.id}/volume/${kind}.nii${q.toString() ? `?${q}` : ''}`;
};
const ahaVersion = c => `${c.apex_slice}-${c.base_first}-${c.model}`;
const volIndex = name => nv.volumes.findIndex(v => v.name === name);

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
      vols.push({ url: volUrl(c, 'aha', ahaVersion(c)), name: 'aha.nii' });
      vols.push({ url: volUrl(c, 'sectors', ahaVersion(c)), name: 'sectors.nii' });
    }
    await nv.loadVolumes(vols);
    applyColormap();
    if (seg) {
      S.ahaKey = ahaVersion(c);
    } else if (S.overlay !== 'none') {
      S.overlay = 'seg';
    }
    shapeCache.clear();
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
  if (!nv.volumes.some(v => v.name === 'aha.nii')) return;
  S.loading = true;
  try {
    for (const kind of ['aha', 'sectors']) {  // both depend on the apex and slice order
      const old = nv.volumes.find(v => v.name === `${kind}.nii`);
      const vol = await niivue.NVImage.loadFromUrl({ url: volUrl(c, kind, ahaVersion(c)), name: `${kind}.nii` });
      if (old) nv.removeVolume(old);
      nv.addVolume(vol);
    }
    S.ahaKey = ahaVersion(c);
    shapeCache.clear();
    applyOverlay();
    applyFrame();
  } finally {
    S.loading = false;
  }
}

function applyOverlay() {
  // the label volumes only provide data (and the readout); drawOverlay draws them
  ['seg.nii', 'aha.nii', 'sectors.nii'].forEach(name => { if (volIndex(name) > 0) nv.setOpacity(volIndex(name), 0); });
  drawOverlay();
}
function applyColormap() {  // the original GUI's gray / viridis choice
  if (!nv.volumes.length) return;
  nv.volumes[0].colormap = S.cmap;
  nv.updateGLVolume();
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

// ---------------------------------------------------------------- overlays
// niivue draws the (interpolated) image; the segmentation, its contours and the
// AHA segments are drawn here as smooth paths: marching squares on a lightly
// blurred mask, then Taubin smoothing and corner cutting. Voxel overlays show the voxel staircase
// once the view is zoomed in; these stay smooth at any size.
const ovCanvas = $('ov');
const ovCtx = ovCanvas.getContext('2d');
const shapeCache = new Map();

function labelSlice(name, k, t) {
  const v = nv.volumes.find(x => x.name === name);
  if (!v) return null;
  const [, nx, ny, nz] = v.hdr.dims;
  const nt = Math.max(1, v.nFrame4D || 1);
  if (k < 0 || k >= nz) return null;
  const n = nx * ny, off = (Math.min(t, nt - 1) * nz + k) * n;
  return { data: v.img.subarray(off, off + n), nx, ny };
}

function blur121(f, w, h) {  // separable [1 2 1]/4, values outside are 0
  const g = new Float32Array(f.length);
  for (let b = 0; b < h; b++) {
    for (let a = 0; a < w; a++) {
      const i = a + b * w;
      g[i] = (2 * f[i] + (a > 0 ? f[i - 1] : 0) + (a < w - 1 ? f[i + 1] : 0)) / 4;
    }
  }
  for (let b = 0; b < h; b++) {
    for (let a = 0; a < w; a++) {
      const i = a + b * w;
      f[i] = (2 * g[i] + (b > 0 ? g[i - w] : 0) + (b < h - 1 ? g[i + w] : 0)) / 4;
    }
  }
}

// Closed iso-lines of f (w x h, zero border) at level L, as [[a, b], ...].
const MS_EDGES = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], null, [[0, 2]], [[3, 2]],
  [[2, 3]], [[0, 2]], null, [[1, 2]], [[3, 1]], [[0, 1]], [[3, 0]], []];
function isoLines(f, w, h, L) {
  const pts = new Map(), nb = new Map();
  const edgeId = (a, b, e) => (e === 0 ? 2 * (a + b * w) : e === 1 ? 2 * (a + 1 + b * w) + 1
    : e === 2 ? 2 * (a + (b + 1) * w) : 2 * (a + b * w) + 1);
  const point = (a, b, e) => {
    const [a0, b0, a1, b1] = e === 0 ? [a, b, a + 1, b] : e === 1 ? [a + 1, b, a + 1, b + 1]
      : e === 2 ? [a, b + 1, a + 1, b + 1] : [a, b, a, b + 1];
    const v0 = f[a0 + b0 * w], v1 = f[a1 + b1 * w];
    const q = (L - v0) / (v1 - v0);
    return [a0 + (a1 - a0) * q, b0 + (b1 - b0) * q];
  };
  const link = (x, y) => {
    (nb.get(x) || nb.set(x, []).get(x)).push(y);
    (nb.get(y) || nb.set(y, []).get(y)).push(x);
  };
  for (let b = 0; b < h - 1; b++) {
    for (let a = 0; a < w - 1; a++) {
      const v00 = f[a + b * w], v10 = f[a + 1 + b * w], v11 = f[a + 1 + (b + 1) * w], v01 = f[a + (b + 1) * w];
      const idx = (v00 > L) | (v10 > L) << 1 | (v11 > L) << 2 | (v01 > L) << 3;
      if (idx === 0 || idx === 15) continue;
      let segs = MS_EDGES[idx];
      if (!segs) {  // saddle: the centre decides which corners connect
        const inside = (v00 + v10 + v11 + v01) / 4 > L;
        segs = (idx === 5) === inside ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]];
      }
      for (const [e0, e1] of segs) {
        const x = edgeId(a, b, e0), y = edgeId(a, b, e1);
        if (!pts.has(x)) pts.set(x, point(a, b, e0));
        if (!pts.has(y)) pts.set(y, point(a, b, e1));
        link(x, y);
      }
    }
  }
  const lines = [], used = new Set();
  for (const start of nb.keys()) {
    if (used.has(start)) continue;
    const line = [];
    let prev = null, cur = start;
    while (cur !== undefined && !used.has(cur)) {
      used.add(cur);
      line.push(pts.get(cur));
      const next = nb.get(cur).find(x => x !== prev && !used.has(x));
      prev = cur; cur = next;
    }
    if (line.length > 2) lines.push(line);
  }
  return lines;
}

function taubin(line, n = 12) {  // removes the voxel ripple without shrinking (closed polyline)
  let p = line;
  for (let it = 0; it < 2 * n; it++) {
    const k = it % 2 ? -0.63 : 0.6;
    p = p.map(([a, b], i) => {
      const [a0, b0] = p[(i + p.length - 1) % p.length], [a1, b1] = p[(i + 1) % p.length];
      return [a + k * ((a0 + a1) / 2 - a), b + k * ((b0 + b1) / 2 - b)];
    });
  }
  return p;
}

function chaikin(line) {  // corner cutting on a closed polyline
  const out = [];
  for (let i = 0; i < line.length; i++) {
    const [a0, b0] = line[i], [a1, b1] = line[(i + 1) % line.length];
    out.push([0.75 * a0 + 0.25 * a1, 0.75 * b0 + 0.25 * b1], [0.25 * a0 + 0.75 * a1, 0.25 * b0 + 0.75 * b1]);
  }
  return out;
}

// Smooth outlines of the voxels passing test, in voxel coordinates (i, j).
function outlines(sl, test) {
  const { data, nx, ny } = sl;
  let i0 = nx, i1 = -1, j0 = ny, j1 = -1;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      if (!test(data[i + j * nx])) continue;
      if (i < i0) i0 = i; if (i > i1) i1 = i;
      if (j < j0) j0 = j; if (j > j1) j1 = j;
    }
  }
  if (i1 < 0) return { lines: [], centre: null };
  const m = 2, w = i1 - i0 + 1 + 2 * m, h = j1 - j0 + 1 + 2 * m;
  const f = new Float32Array(w * h);
  let si = 0, sj = 0, n = 0;
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      if (!test(data[i + j * nx])) continue;
      f[i - i0 + m + (j - j0 + m) * w] = 1;
      si += i; sj += j; n++;
    }
  }
  blur121(f, w, h);
  const lines = isoLines(f, w, h, 0.5).map(l => chaikin(chaikin(taubin(l))).map(([a, b]) => [a + i0 - m, b + j0 - m]));
  return { lines, centre: [si / n, sj / n] };
}

function shapesAt(mode, k, t) {
  const key = `${mode}|${k}|${t}`;
  if (shapeCache.has(key)) return shapeCache.get(key);
  let out = null;
  if (mode === 'aha' || mode === 'sectors') {
    const sl = labelSlice(`${mode}.nii`, k, t);
    if (sl) {
      const present = [...new Set(sl.data)].filter(v => v >= 1 && v <= 17).sort((x, y) => x - y);
      out = { segments: present.map(sgm => ({ sgm, ...outlines(sl, v => v === sgm) })) };
    }
  } else {
    const sl = labelSlice('seg.nii', k, t);
    if (sl) {
      out = {
        endo: outlines(sl, v => v === 1).lines,
        epi: outlines(sl, v => v === 1 || v === 2).lines,
        rv: outlines(sl, v => v === 3).lines,
      };
    }
  }
  if (shapeCache.size > 3000) shapeCache.clear();
  shapeCache.set(key, out);
  return out;
}

function drawOverlay() {
  const gl = $('gl');
  if (ovCanvas.width !== gl.width || ovCanvas.height !== gl.height) {
    ovCanvas.width = gl.width; ovCanvas.height = gl.height;
  }
  const ctx = ovCtx;
  ctx.clearRect(0, 0, ovCanvas.width, ovCanvas.height);
  const c = selectedCase();
  const mode = S.overlay;
  if (!S.loadedKey || mode === 'none' || !isSegmented(c) || volIndex('seg.nii') < 0) return renderLegend();
  const [dx, dy] = c.zooms || [1, 1];
  const nz = nv.volumes[0].hdr.dims[3];
  const dpr = nv.uiData.dpr || 1;
  const walls = new Set();
  for (const tile of nv.screenSlices) {
    if (tile.axCorSag !== 0 || !tile.fovMM || tile.fovMM.length < 2) continue;
    const k = Math.round(tile.sliceFrac * nz - 0.5);
    const g = shapesAt(mode === 'aha' || mode === 'sectors' ? mode : 'seg', k, S.frame);
    if (!g) continue;
    const [L, T, W, H] = tile.leftTopWidthHeight;
    const [mx, my] = tile.leftTopMM, [fx, fy] = tile.fovMM;
    // voxel (i, j) -> mm (dy * j, -dx * i), the display affine -> canvas pixels
    const X = j => L + (dy * j - mx) / fx * W, Y = i => T + (1 - (-dx * i - my) / fy) * H;
    const path = (lines, p = new Path2D()) => {
      for (const l of lines) {
        l.forEach(([i, j], n) => (n ? p.lineTo(X(j), Y(i)) : p.moveTo(X(j), Y(i))));
        p.closePath();
      }
      return p;
    };
    const big = W > 360 * dpr;
    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, W, H); ctx.clip();
    ctx.lineJoin = 'round';
    if (mode === 'edge') {
      ctx.lineWidth = (big ? 1.6 : 1.2) * dpr;
      [['epi', SEG_LINE.MYO], ['endo', SEG_LINE.LV], ['rv', SEG_LINE.RV]].forEach(([name, col]) => {
        ctx.strokeStyle = rgb(col); ctx.stroke(path(g[name]));
      });
    } else if (mode === 'lv' || mode === 'rv') {  // one structure, as the original LV / RV views
      const p = path(mode === 'lv' ? g.endo : g.rv), name = mode === 'lv' ? 'LV' : 'RV';
      ctx.fillStyle = rgb(SEG_FILL[name], Math.min(1, S.opacity + 0.2)); ctx.fill(p);
      ctx.lineWidth = 1 * dpr; ctx.strokeStyle = rgb(SEG_LINE[name], 0.9); ctx.stroke(p);
    } else if (mode === 'seg') {
      const myo = path(g.endo, path(g.epi)), lv = path(g.endo), rv = path(g.rv);
      ctx.fillStyle = rgb(SEG_FILL.MYO, S.opacity); ctx.fill(myo, 'evenodd');
      ctx.fillStyle = rgb(SEG_FILL.LV, S.opacity); ctx.fill(lv);
      ctx.fillStyle = rgb(SEG_FILL.RV, S.opacity); ctx.fill(rv);
      ctx.lineWidth = 1 * dpr;
      [[path(g.epi), SEG_LINE.MYO], [lv, SEG_LINE.LV], [rv, SEG_LINE.RV]].forEach(([p, col]) => {
        ctx.strokeStyle = rgb(col, 0.85); ctx.stroke(p);
      });
    } else {
      // 6/4 sectors (original GUI): sector n has the wall of AHA segment n, or
      // of 12 + n on the apical slices that have 4 sectors
      const apical = mode === 'sectors' && c.apex_slice !== null &&
        (c.base_first === false ? k <= c.apex_slice : k >= c.apex_slice);
      const wallOf = n => AHA_WALL[mode === 'aha' ? n : (apical ? 12 + Math.min(n, 4) : n)];
      if (mode === 'sectors' && g.segments.length) walls.add(apical ? 'n4' : 'n6');
      for (const sg of g.segments) {
        const col = WALL_COLORS[wallOf(sg.sgm)];
        walls.add(wallOf(sg.sgm));
        ctx.fillStyle = rgb(col, Math.min(1, S.opacity + 0.2)); ctx.fill(path(sg.lines));
      }
      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      for (const sg of g.segments) ctx.stroke(path(sg.lines));
      if (big) {
        ctx.font = `600 ${12 * dpr}px system-ui, sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.lineWidth = 3 * dpr; ctx.strokeStyle = 'rgba(0,0,0,0.75)'; ctx.fillStyle = '#fff';
        for (const sg of g.segments) {
          const [i, j] = sg.centre;
          ctx.strokeText(sg.sgm, X(j), Y(i)); ctx.fillText(sg.sgm, X(j), Y(i));
        }
      }
    }
    ctx.restore();
  }
  if (mode === 'aha' || mode === 'sectors') {
    const counts = ['n6', 'n4'].filter(n => walls.has(n)).map(n => t('sectorsLegend', n.slice(1))).join(' / ');
    renderLegend(Object.keys(WALL_COLORS).filter(w => walls.has(w)).map(w => [WALL_COLORS[w], t('wall_' + w)])
      .concat(counts ? [[null, counts]] : []));
  } else if (mode === 'lv' || mode === 'rv') {
    renderLegend([mode === 'lv' ? [SEG_FILL.LV, t('lblLV')] : [SEG_FILL.RV, t('lblRV')]]);
  } else if (mode === 'edge') {
    renderLegend([[SEG_LINE.LV, t('lblEndo')], [SEG_LINE.MYO, t('lblEpi')], [SEG_LINE.RV, t('lblRV')]]);
  } else {
    renderLegend([[SEG_FILL.LV, t('lblLV')], [SEG_FILL.MYO, t('lblMYO')], [SEG_FILL.RV, t('lblRV')]]);
  }
}
const drawScene = nv.drawScene.bind(nv);
nv.drawScene = (...args) => { const r = drawScene(...args); drawOverlay(); return r; };

// ---------------------------------------------------------------- actions
async function addPaths(paths, include = '', exclude = '', results = false) {
  const r = await api('/api/cases/add', { paths, include, exclude, results });
  say(t('addedFull', r.added, r.updated || 0, (r.missing || []).length), (r.missing || []).length > 0);
  await refresh();
}
// native file dialogs: the Rust app's bridge or the pywebview API
const native = () => window.tigerhxNative || (window.pywebview && window.pywebview.api);

function showAddPanel(paths = []) {
  $('addPaths').value = paths.join('\n');
  $('addPanel').classList.remove('hidden');
  $('addPaths').focus();
}

function bind() {
  $('btnAddFiles').onclick = guard(async () => {
    if (!native()) return showAddPanel();
    const files = await native().pick_files();
    if (files && files.length) await addPaths(files);
  });
  $('btnAddFolder').onclick = guard(async () => {
    if (!native()) return showAddPanel();
    const dirs = await native().pick_folder();
    if (dirs && dirs.length) showAddPanel(dirs);
  });
  $('btnAddConfirm').onclick = guard(async () => {
    const paths = $('addPaths').value.split('\n').map(s => s.trim()).filter(Boolean);
    if (!paths.length) return;
    await addPaths(paths, $('addInclude').value, $('addExclude').value);
    $('addPanel').classList.add('hidden');
  });
  $('btnAddCancel').onclick = () => $('addPanel').classList.add('hidden');
  $('btnGenCsv').onclick = guard(async () => {
    const paths = $('addPaths').value.split('\n').map(s => s.trim()).filter(Boolean);
    if (!paths.length) return;
    const r = await api('/api/gencsv', { paths, include: $('addInclude').value, exclude: $('addExclude').value });
    say(t('genCsvDone', r.count, r.path));
    $('addPanel').classList.add('hidden');
  });
  $('btnOpenResults').onclick = guard(async () => {
    let dirs = null;
    if (native()) dirs = await native().pick_folder();
    else {
      const d = window.prompt(t('resultsPrompt'), S.state ? S.state.output_dir : '');
      dirs = d ? [d] : null;
    }
    if (dirs && dirs.length) await addPaths(dirs, '', '', true);
  });
  $('outputDir').onchange = guard(async () => {
    await api('/api/settings', { output_dir: $('outputDir').value });
    await refresh();
  });
  $('btnOpenOutput').onclick = guard(() => api('/api/open', { folder: 'output' }));
  $('btnOpenCsv').onclick = guard(() => api('/api/open', { folder: 'csv' }));
  $('btnDownloadAll').onclick = guard(async () => {
    const r = await api('/api/models/download', {});
    say(t('downloadStarted', r.downloading));
    await refresh();
  });
  $('btnLog').onclick = () => {
    S.logOpen = !S.logOpen;
    $('logPanel').classList.toggle('hidden', !S.logOpen);
    $('btnLog').classList.remove('alert');
    if (S.logOpen) $('logBody').scrollTop = $('logBody').scrollHeight;
  };
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
    drawOverlay();
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
  $('overlayMode').onchange = () => { S.overlay = $('overlayMode').value; applyOverlay(); renderViewerBars(selectedCase()); };
  $('cmap').onchange = () => {
    S.cmap = $('cmap').value;
    try { localStorage.setItem('tigerhx_cmap', S.cmap); } catch (e) { /* ignore */ }
    applyColormap();
  };
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
