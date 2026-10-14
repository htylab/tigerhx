'use strict';

// ---------------------------------------------------------------- strings
const I18N = {
  zh: {
    addFiles: '＋ 檔案', addFolder: '＋ 資料夾', model: '模型', stop: '■ 停止',
    runAll: n => `▶ 分割待處理 (${n})`, runChecked: n => `▶ 分割已選 (${n})`,
    export: '匯出指標 CSV', settings: '設定', pathsLabel: '檔案或資料夾路徑（每行一個）',
    include: '包含關鍵字（逗號分隔，空白=全部）', exclude: '排除關鍵字', add: '加入', cancel: '取消',
    saveMat: '在工作資料夾輸出 .mat（舊版格式）', matInput: '.mat 也存到原始資料夾',
    niiInput: '分割 .nii.gz 存到原始資料夾', workspace: '工作資料夾', openFolder: '開啟資料夾',
    cases: '病例', colName: '名稱', colStatus: '狀態', colApex: 'Apex',
    emptyCases: '按「＋ 檔案」或「＋ 資料夾」加入 cine NIfTI。\n不需要先準備 CSV，也不用先填 Apex：分割完成後再看著影像確認。',
    remove: '移除選取', openCaseFolder: '開啟病例資料夾',
    viewSingle: '單張', viewMosaic: '全部 slice', ovSeg: '分割', ovAha: 'AHA 17', ovNone: '無疊圖',
    opacity: '透明度', hintSelect: '從左側選一個病例', frame: 'Frame', slice: 'Slice',
    apexStart: 'Apical 起始 slice', setApex: '把目前 slice 設為 Apex', confirmApex: '確認建議值',
    sliceOrder: 'Slice 順序', baseFirst: 'base → apex', apexFirst: 'apex → base',
    results: '結果', noResults: '分割完成後會顯示容積與 EF。',
    st_pending: '待處理', st_queued: '排隊中', st_running: '分割中', st_review: '待確認 Apex',
    st_done: '完成', st_error: '錯誤',
    suggested: '自動建議', confirmed: '已確認',
    lv_basal: 'basal', lv_mid: 'mid', lv_apical: 'apical', lv_apex: 'apex cap',
    added: n => `加入 ${n} 個病例`, queued: n => `${n} 個病例排入佇列`, exported: p => `已匯出：${p}`,
    downloading: (m, p) => `下載模型 ${m} … ${p}`, runningCase: (n, p) => `分割中：${n} ${p}`,
    queueLeft: n => `，佇列剩 ${n}`, idle: '待命', apexSaved: k => `Apex 設為 slice ${k}，AHA 已重算`,
    confirmRemove: n => `從清單移除 ${n} 個病例？（不會刪除檔案）`,
    lvEf: 'LV EF', rvEf: 'RV EF', volCurve: '容積–時間曲線 (mL)', table: '數值表',
    edv: 'EDV', esv: 'ESV', sv: 'SV', mass: '心肌質量', edFrame: 'ED frame', esFrame: 'ES frame',
    singleFrame: '單一時相：無 EF。', notSegmented: '尚未分割。', error: '錯誤',
    lblLV: 'LV 血池', lblMYO: '心肌', lblRV: 'RV 血池', basal: 'Basal', mid: 'Mid', apical: 'Apical', apex: 'Apex',
    modelMissing: '（首次使用時下載）',
    calib: 'ED / ES 與 slice 校正', auto: '自動', manual: '手動', resetPhases: '恢復自動 ED/ES',
    useFrame: f => `用目前 frame（${f}）`, setSlice: k => `設為目前 slice（${k}）`, clear: '清除', all: '全部',
    setTo: v => `設為 ${v}`,
    exclTitle: '不計入容積的 slice', checksTitle: '建議檢查',
    dontCount: ch => `此 slice 不計入 ${ch}`, countAgain: ch => `恢復計入 ${ch}`,
    atSlice: (ph, f, k) => `目前 ${ph}（frame ${f}）· slice ${k}`,
    goPhase: '切到 ED 或 ES 才能排除 slice：', show: '查看',
    chkBigger: (z, ch, es, ed) => `slice ${z}：ES 的 ${ch} 比 ED 大（${es} vs ${ed} voxels），可能標到心室外的結構`,
    chkGap: (z, ph, ch, a, n) => `slice ${z}：${ph} 的 ${ch} 疑似漏標（${a} voxels，鄰近 ${n}）`,
    chkBasal: (z, ch) => `確認 ES 最基底的 ${ch}（slice ${z}）是否已進入心房；是的話不計入`,
    edRule: 'ED 自動判定', edRuleMax: '最大 LV 容積', edRuleFirst: '第一個 frame（R 波觸發）',
    rule_max_volume: '最大容積', rule_first_frame: '第一個 frame', rule_min: '最小容積',
    excluded: (ph, ch) => `${ph} 不計入 ${ch}`, curveNote: '曲線包含所有 slice，未套用排除設定。',
    now: (f, k) => `目前：frame ${f} · slice ${k}`,
  },
  en: {
    addFiles: '＋ Files', addFolder: '＋ Folder', model: 'Model', stop: '■ Stop',
    runAll: n => `▶ Segment pending (${n})`, runChecked: n => `▶ Segment checked (${n})`,
    export: 'Export metrics CSV', settings: 'Settings', pathsLabel: 'File or folder paths (one per line)',
    include: 'Include keywords (comma separated, empty = all)', exclude: 'Exclude keywords', add: 'Add', cancel: 'Cancel',
    saveMat: 'Write legacy .mat to the workspace', matInput: 'Also write .mat next to the input',
    niiInput: 'Write segmentation .nii.gz next to the input', workspace: 'Workspace', openFolder: 'Open folder',
    cases: 'Cases', colName: 'Name', colStatus: 'Status', colApex: 'Apex',
    emptyCases: 'Use “＋ Files” or “＋ Folder” to add cine NIfTI images.\nNo CSV and no apex needed up front: confirm the apex on the images after segmentation.',
    remove: 'Remove checked', openCaseFolder: 'Open case folder',
    viewSingle: 'Single', viewMosaic: 'All slices', ovSeg: 'Segmentation', ovAha: 'AHA 17', ovNone: 'No overlay',
    opacity: 'Opacity', hintSelect: 'Select a case on the left', frame: 'Frame', slice: 'Slice',
    apexStart: 'First apical slice', setApex: 'Use current slice as apex', confirmApex: 'Confirm suggestion',
    sliceOrder: 'Slice order', baseFirst: 'base → apex', apexFirst: 'apex → base',
    results: 'Results', noResults: 'Volumes and EF appear after segmentation.',
    st_pending: 'Pending', st_queued: 'Queued', st_running: 'Segmenting', st_review: 'Check apex',
    st_done: 'Done', st_error: 'Error',
    suggested: 'suggested', confirmed: 'confirmed',
    lv_basal: 'basal', lv_mid: 'mid', lv_apical: 'apical', lv_apex: 'apex cap',
    added: n => `Added ${n} case(s)`, queued: n => `Queued ${n} case(s)`, exported: p => `Exported: ${p}`,
    downloading: (m, p) => `Downloading model ${m} … ${p}`, runningCase: (n, p) => `Segmenting: ${n} ${p}`,
    queueLeft: n => `, ${n} queued`, idle: 'Idle', apexSaved: k => `Apex set to slice ${k}; AHA updated`,
    confirmRemove: n => `Remove ${n} case(s) from the list? (files are kept)`,
    lvEf: 'LV EF', rvEf: 'RV EF', volCurve: 'Volume–time curve (mL)', table: 'Values',
    edv: 'EDV', esv: 'ESV', sv: 'SV', mass: 'Myocardial mass', edFrame: 'ED frame', esFrame: 'ES frame',
    singleFrame: 'Single frame: no EF.', notSegmented: 'Not segmented yet.', error: 'Error',
    lblLV: 'LV blood', lblMYO: 'Myocardium', lblRV: 'RV blood', basal: 'Basal', mid: 'Mid', apical: 'Apical', apex: 'Apex',
    modelMissing: ' (downloaded on first use)',
    calib: 'ED / ES and slices', auto: 'auto', manual: 'manual', resetPhases: 'Automatic ED/ES',
    useFrame: f => `Use current frame (${f})`, setSlice: k => `Set to current slice (${k})`, clear: 'Clear', all: 'all',
    setTo: v => `Set ${v}`,
    exclTitle: 'Slices not counted', checksTitle: 'Worth checking',
    dontCount: ch => `Don't count this slice for ${ch}`, countAgain: ch => `Count for ${ch} again`,
    atSlice: (ph, f, k) => `Now ${ph} (frame ${f}) · slice ${k}`,
    goPhase: 'Go to ED or ES to exclude a slice:', show: 'Show',
    chkBigger: (z, ch, es, ed) => `slice ${z}: ${ch} larger at ES than at ED (${es} vs ${ed} voxels), possibly a structure outside the ventricle`,
    chkGap: (z, ph, ch, a, n) => `slice ${z}: ${ch} possibly missed at ${ph} (${a} voxels, neighbours ${n})`,
    chkBasal: (z, ch) => `Check whether the basal-most ${ch} slice at ES (slice ${z}) is already atrium; if so, don't count it`,
    edRule: 'Automatic ED', edRuleMax: 'largest LV volume', edRuleFirst: 'first frame (R-wave triggered)',
    rule_max_volume: 'max volume', rule_first_frame: 'first frame', rule_min: 'min volume',
    excluded: (ph, ch) => `${ch} not counted at ${ph}`, curveNote: 'Curves include all slices; exclusions are not applied.',
    now: (f, k) => `Now: frame ${f} · slice ${k}`,
  },
};
let lang = 'zh';
try { lang = localStorage.getItem('tigerhx_lang') || 'zh'; } catch (e) { /* storage blocked */ }
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
  view: 'single', overlay: 'seg', opacity: 0.45,
  frame: 0, slice: null, playing: null,
  loadedKey: null, ahaKey: null, loading: false, message: '', error: false,
};
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined ? '–' : Number(v).toFixed(d));

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
const isSegmented = c => c && ['review', 'done'].includes(c.status) && c.metrics;

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
  $('gpu').checked = !!st.settings.gpu;
  document.querySelectorAll('[data-setting]').forEach(el => { el.checked = !!st.settings[el.dataset.setting]; });
  if (document.activeElement !== $('edRule')) $('edRule').value = st.settings.ed_rule || 'max_volume';
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
    const ef = c.metrics ? fmt(c.metrics.LV_EF_pct) : '–';
    const nWarn = c.metrics ? openChecks(c).length : 0;
    const warn = nWarn ? ` <span class="warn-mark" title="${esc(t('checksTitle'))}: ${nWarn}">⚠${nWarn}</span>` : '';
    const msg = c.message && c.status === 'error' ? `<div class="case-msg">${esc(c.message)}</div>` : '';
    return `<tr class="row${c.id === S.selected ? ' selected' : ''}" data-id="${c.id}" title="${esc(c.path)}">
      <td><input type="checkbox" data-check="${c.id}"${S.checked.has(c.id) ? ' checked' : ''}></td>
      <td class="name">${esc(c.name)}${msg}</td><td>${chip(c)}</td>
      <td class="num">${esc(apex)}</td><td class="num">${esc(ef)}${warn}</td></tr>`;
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
  if (seg && S.slice !== null) {
    [['ED', c.ed], ['ES', c.es]].forEach(([ph, f]) => {
      if (f !== S.frame) return;
      ['LV', 'RV'].forEach(ch => { if (isExcluded(c, ph, ch, S.slice)) notes.push(t('excluded', ph, ch)); });
    });
  }
  $('levelTag').classList.toggle('hidden', !notes.length);
  $('levelTag').classList.toggle('warn', notes.length > 1);
  $('levelTag').textContent = notes.join(' · ');
  renderCalib(c);

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
  document.querySelectorAll('#overlayMode button').forEach(b => {
    b.classList.toggle('on', b.dataset.v === S.overlay);
    b.disabled = b.dataset.v !== 'none' && !seg;
  });
}

// ---------------------------------------------------------------- results
function renderResults(c) {
  const box = $('results');
  const key = c ? JSON.stringify([c.id, c.status, c.metrics, c.ed, c.es, c.excluded, lang, c.message]) : 'none';
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  if (!c) { box.innerHTML = `<div class="empty">${esc(t('noResults'))}</div>`; return; }
  if (!isSegmented(c)) {
    const msg = c.status === 'error' ? `<div class="case-msg">${esc(c.message)}</div>` : '';
    box.innerHTML = `<div class="info"><b>${esc(c.name)}</b><br><span class="mono">${esc(c.path)}</span></div>
      <div class="empty">${esc(t('notSegmented'))}</div>${msg}`;
    return;
  }
  const m = c.metrics;
  const multi = c.shape[3] > 1;
  const row = (k, lv, rv, unit, d = 1) =>
    `<tr><th>${esc(k)} <span class="muted">${unit}</span></th><td>${fmt(lv, d)}</td><td>${fmt(rv, d)}</td></tr>`;
  box.innerHTML = `
    <div class="hero">
      <div class="tile"><div class="k">${t('lvEf')}</div><div class="v">${fmt(m.LV_EF_pct)}<small>%</small></div></div>
      <div class="tile"><div class="k">${t('rvEf')}</div><div class="v">${fmt(m.RV_EF_pct)}<small>%</small></div></div>
    </div>
    ${multi ? '' : `<div class="info">${esc(t('singleFrame'))}</div>`}
    <table class="metrics">
      <thead><tr><th></th><th style="text-align:right">LV</th><th style="text-align:right">RV</th></tr></thead>
      <tbody>
        ${row(t('edv'), m.LV_EDV_mL, m.RV_EDV_mL, 'mL')}
        ${row(t('esv'), m.LV_ESV_mL, m.RV_ESV_mL, 'mL')}
        ${row(t('sv'), m.LV_SV_mL, m.RV_SV_mL, 'mL')}
        <tr><th>${esc(t('mass'))} <span class="muted">g</span></th><td>${fmt(m.LV_mass_g)}</td><td>–</td></tr>
      </tbody>
    </table>
    ${multi ? '<div id="calib" class="chart-card"></div>' : ''}
    <div class="info">${esc(c.model)}<br><span class="mono">${esc(c.path)}</span></div>
    ${multi ? `<div class="chart-card">
      <div class="chart-title">${esc(t('volCurve'))}
        <span class="key"><i style="background:var(--series-lv)"></i>LV</span>
        <span class="key"><i style="background:var(--series-rv)"></i>RV</span></div>
      <div class="chart" id="chart"></div>
      ${hasExclusions(c) ? `<div class="info">${esc(t('curveNote'))}</div>` : ''}
      <details><summary>${esc(t('table'))}</summary><div id="curveTable"></div></details>
    </div>` : ''}`;
  if (multi) { drawChart(c); renderCalib(c, true); }
}

const hasExclusions = c => ['ED', 'ES'].some(ph => ['LV', 'RV'].some(ch => c.excluded[ph][ch].length));
const isExcluded = (c, phase, chamber, k) => !!(c.excluded && c.excluded[phase][chamber].includes(k));
const phaseOfFrame = c => (S.frame === c.ed ? 'ED' : S.frame === c.es ? 'ES' : null);
const openChecks = c => (c.checks || []).filter(k => k.kind !== 'basal' && !isExcluded(c, k.phase, k.chamber, k.slice));

function checkText(k) {
  if (k.kind === 'es_gt_ed') return t('chkBigger', k.slice, k.chamber, k.es, k.ed);
  if (k.kind === 'gap') return t('chkGap', k.slice, k.phase, k.chamber, k.area, k.neighbours);
  return t('chkBasal', k.slice, k.chamber);
}

function renderCalib(c, force = false) {
  const box = $('calib');
  if (!box || !isSegmented(c) || c.shape[3] < 2) return;
  const key = JSON.stringify([c.id, c.ed, c.es, c.ed_manual, c.es_manual, c.excluded, c.checks, S.frame, S.slice,
    lang, S.state.settings.ed_rule]);
  if (!force && box.dataset.key === key) return;
  box.dataset.key = key;
  const rule = S.state.settings.ed_rule || 'max_volume';
  const src = (manual, autoRule) => `<span class="tag${manual === null ? '' : ' ok'}">${esc(manual === null
    ? `${t('auto')} · ${t('rule_' + autoRule)}` : t('manual'))}</span>`;
  const phase = phaseOfFrame(c);
  const k = S.slice;

  let toggles;
  if (phase && k !== null) {
    toggles = ['LV', 'RV'].map(ch => {
      const on = isExcluded(c, phase, ch, k);
      return `<button class="small${on ? ' on' : ''}" data-exclude="${phase},${ch},${k},${on ? 0 : 1}">
        ${esc(on ? t('countAgain', ch) : t('dontCount', ch))}</button>`;
    }).join(' ');
    toggles = `<div class="info">${esc(t('atSlice', phase, S.frame, k))}</div><div>${toggles}</div>`;
  } else {
    toggles = `<div class="info">${esc(t('goPhase'))}
      <button class="small" data-goto="${c.ed},${k ?? ''}">ED</button>
      <button class="small" data-goto="${c.es},${k ?? ''}">ES</button></div>`;
  }

  const lists = ['ED', 'ES'].map(ph => ['LV', 'RV'].map(ch => c.excluded[ph][ch].map(z =>
    `<span class="tag ok">${ph} ${ch} · slice ${z}
      <button class="chip-x" data-exclude="${ph},${ch},${z},0" title="${esc(t('countAgain', ch))}">×</button></span>`
  ).join(' ')).join(' ')).join(' ').trim();

  const checks = (c.checks || []).map(chk => {
    const done = chk.kind !== 'basal' && isExcluded(c, chk.phase, chk.chamber, chk.slice);
    const icon = chk.kind === 'basal' ? 'ⓘ' : done ? '✓' : '⚠';
    const f = chk.phase === 'ED' ? c.ed : c.es;
    return `<li class="${chk.kind === 'basal' ? 'note' : done ? 'done' : 'warn'}"><span class="ico">${icon}</span>
      <span>${esc(checkText(chk))}</span>
      <button class="small" data-goto="${f},${chk.slice}">${esc(t('show'))}</button></li>`;
  }).join('');

  box.innerHTML = `
    <div class="chart-title">${esc(t('calib'))}</div>
    <table class="metrics calib">
      <tr><th>ED</th><td>frame <b>${c.ed}</b> ${src(c.ed_manual, rule)}</td>
        <td><button class="small" data-phase="ed" title="${esc(t('useFrame', S.frame))}">${esc(t('setTo', S.frame))}</button></td></tr>
      <tr><th>ES</th><td>frame <b>${c.es}</b> ${src(c.es_manual, 'min')}</td>
        <td><button class="small" data-phase="es" title="${esc(t('useFrame', S.frame))}">${esc(t('setTo', S.frame))}</button></td></tr>
    </table>
    ${c.ed_manual !== null || c.es_manual !== null
      ? `<button class="small" data-phase="reset">${esc(t('resetPhases'))}</button>` : ''}
    <div class="sub-title">${esc(t('exclTitle'))}</div>
    ${toggles}
    ${lists ? `<div class="excl-list">${lists}</div>` : ''}
    ${checks ? `<div class="sub-title">${esc(t('checksTitle'))}</div><ul class="checks">${checks}</ul>` : ''}`;
}

async function gotoFrameSlice(f, k) {
  if (S.view === 'mosaic') { S.view = 'single'; applyView(); }
  setFrame(f);
  if (k !== '' && k !== undefined && k !== null) goToSlice(+k);
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  return [1, 2, 2.5, 5, 10].map(f => f * p).find(x => x >= v);
}

function drawChart(c) {
  const host = $('chart');
  const lv = c.curves.LV, rv = c.curves.RV;
  const n = lv.length;
  const W = Math.max(240, host.clientWidth || 280), H = 170;
  const m = { l: 34, r: 24, t: 18, b: 22 };
  const ymax = niceMax(Math.max(...lv, ...rv) * 1.05);
  const x = i => m.l + (n === 1 ? 0 : i / (n - 1)) * (W - m.l - m.r);
  const y = v => H - m.b - v / ymax * (H - m.t - m.b);
  const path = arr => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const ticks = [0, 0.5, 1].map(f => f * ymax);
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', t('volCurve'));
  const add = (tag, attrs, text) => {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    if (text !== undefined) el.textContent = text;
    svg.appendChild(el);
    return el;
  };
  ticks.forEach(v => {
    add('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1 });
    add('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 10, fill: 'var(--muted)' }, Math.round(v));
  });
  add('line', { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: 'var(--axis)', 'stroke-width': 1 });
  const anchor = f => (f === 0 ? 'start' : f === n - 1 ? 'end' : 'middle');
  [[c.ed, 'ED'], [c.es, 'ES']].forEach(([f, label]) => {
    add('line', { x1: x(f), x2: x(f), y1: m.t, y2: H - m.b, stroke: 'var(--axis)', 'stroke-width': 1 });
    add('text', { x: x(f) + (f === 0 ? 3 : f === n - 1 ? -3 : 0), y: m.t - 4, 'text-anchor': anchor(f), 'font-size': 10, fill: 'var(--ink-2)' }, label);
  });
  [0, n - 1].forEach(f => add('text', { x: x(f), y: H - 6, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--muted)' }, f));
  const cur = add('line', { x1: x(S.frame), x2: x(S.frame), y1: m.t, y2: H - m.b, stroke: 'var(--ink-2)', 'stroke-width': 1, 'stroke-dasharray': '' });
  add('path', { d: path(lv), fill: 'none', stroke: 'var(--series-lv)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  add('path', { d: path(rv), fill: 'none', stroke: 'var(--series-rv)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  if (Math.abs(y(lv[n - 1]) - y(rv[n - 1])) >= 12) {  // converging ends: legend + tooltip only
    add('text', { x: x(n - 1) + 4, y: y(lv[n - 1]) + 4, 'font-size': 10, fill: 'var(--ink-2)' }, 'LV');
    add('text', { x: x(n - 1) + 4, y: y(rv[n - 1]) + 4, 'font-size': 10, fill: 'var(--ink-2)' }, 'RV');
  }
  const hover = add('line', { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-width': 1, visibility: 'hidden' });
  host.innerHTML = '';
  host.appendChild(svg);
  const tip = document.createElement('div');
  tip.className = 'tip hidden';
  host.appendChild(tip);

  const frameAt = ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * W / r.width;
    return Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
  };
  svg.addEventListener('pointermove', ev => {
    const f = frameAt(ev);
    hover.setAttribute('x1', x(f)); hover.setAttribute('x2', x(f));
    hover.setAttribute('visibility', 'visible');
    tip.replaceChildren();
    const head = document.createElement('div');
    head.textContent = `Frame ${f}${f === c.ed ? ' · ED' : ''}${f === c.es ? ' · ES' : ''}`;
    tip.appendChild(head);
    [['LV', lv[f], '--series-lv'], ['RV', rv[f], '--series-rv']].forEach(([name, v, col]) => {
      const r = document.createElement('div');
      r.className = 'tip-row';
      const k = document.createElement('i'); k.style.background = `var(${col})`;
      const b = document.createElement('b'); b.textContent = `${fmt(v)} mL`;
      const s = document.createElement('span'); s.className = 'muted'; s.textContent = name;
      r.append(k, b, s);
      tip.appendChild(r);
    });
    tip.classList.remove('hidden');
    const rect = svg.getBoundingClientRect();
    const left = x(f) * rect.width / W;
    tip.style.left = `${Math.min(left + 8, rect.width - tip.offsetWidth - 4)}px`;
    tip.style.top = '4px';
  });
  svg.addEventListener('pointerleave', () => { hover.setAttribute('visibility', 'hidden'); tip.classList.add('hidden'); });
  svg.addEventListener('click', ev => setFrame(frameAt(ev)));
  S.chartCursor = f => { cur.setAttribute('x1', x(f)); cur.setAttribute('x2', x(f)); };

  const tbl = $('curveTable');
  const rows = lv.map((v, i) => `<tr><th>${i}</th><td>${fmt(v)}</td><td>${fmt(c.curves.MYO[i])}</td><td>${fmt(rv[i])}</td></tr>`).join('');
  tbl.innerHTML = `<table class="metrics"><thead><tr><th>Frame</th><th style="text-align:right">LV</th>
    <th style="text-align:right">${esc(t('lblMYO'))}</th><th style="text-align:right">RV</th></tr></thead><tbody>${rows}</tbody></table>`;
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
  if (S.overlay === 'seg') {
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
  if (!S.loadedKey || S.overlay === 'none' || nv.volumes.length < 2) { el.classList.add('hidden'); return; }
  const items = S.overlay === 'seg'
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

const volUrl = (c, kind, v = '') => `/api/cases/${c.id}/volume/${kind}.nii${v ? `?v=${encodeURIComponent(v)}` : ''}`;
const ahaVersion = c => `${c.apex_slice}-${c.base_first}-${c.model}`;

async function maybeLoadCase(c) {
  const seg = isSegmented(c);
  const key = `${c.id}|${seg ? c.model : 'raw'}`;
  if (S.loading) return;
  if (key === S.loadedKey) {
    if (seg && ahaVersion(c) !== S.ahaKey) await reloadAha(c);
    return;
  }
  S.loading = true;
  try {
    $('canvasHint').textContent = '…';
    $('canvasHint').classList.remove('hidden');
    stopPlay();
    const vols = [{ url: volUrl(c, 'image'), name: 'image.nii' }];
    if (seg) {
      vols.push({ url: volUrl(c, 'seg', Date.now()), name: 'seg.nii' });
      vols.push({ url: volUrl(c, 'aha', ahaVersion(c)), name: 'aha.nii' });
    }
    await nv.loadVolumes(vols);
    if (seg) {
      nv.volumes[1].setColormapLabel(lut(SEG_COLORS, ['', 'LV', 'MYO', 'RV']));
      nv.volumes[2].setColormapLabel(lut(ahaColors(), AHA_NAMES.map((n, i) => (i ? `${i}` : ''))));
      S.ahaKey = ahaVersion(c);
    } else if (S.overlay !== 'none') {
      S.overlay = 'seg';
    }
    S.loadedKey = key;
    const nz = c.shape ? c.shape[2] : nv.volumes[0].dims[3];
    S.frame = seg && c.ed !== null ? c.ed : 0;
    S.slice = seg && c.levels ? c.levels.findIndex(l => l === 'mid') : Math.floor(nz / 2);
    if (S.slice < 0) S.slice = Math.floor(nz / 2);
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
  if (nv.volumes.length < 3) return;
  S.loading = true;
  try {
    const old = nv.volumes[2];
    const vol = await niivue.NVImage.loadFromUrl({ url: volUrl(c, 'aha', ahaVersion(c)), name: 'aha.nii' });
    vol.setColormapLabel(lut(ahaColors(), AHA_NAMES.map((n, i) => (i ? `${i}` : ''))));
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
  if (nv.volumes.length >= 3) {
    nv.setOpacity(1, S.overlay === 'seg' ? S.opacity : 0);
    nv.setOpacity(2, S.overlay === 'aha' ? S.opacity : 0);
  }
  renderLegend();
}
function applyFrame() {
  nv.volumes.forEach(v => { if (v.nFrame4D > 1) nv.setFrame4D(v.id, S.frame); });
  if (S.chartCursor) S.chartCursor(S.frame);
}
function applyView() {
  const c = selectedCase();
  if (S.view === 'mosaic' && nv.volumes.length) {
    const [nx, ny, nz] = [c.shape[0], c.shape[1], c.shape[2]];
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
  const nz = c.shape ? c.shape[2] : nv.volumes[0].dims[3];
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
  $('gpu').onchange = guard(async () => { await api('/api/settings', { gpu: $('gpu').checked }); await refresh(); });
  $('btnRun').onclick = guard(async () => {
    const ids = [...S.checked].filter(id => S.state.cases.some(c => c.id === id));
    const r = await api('/api/run', { ids });
    say(t('queued', r.queued));
    S.checked.clear();
    await refresh();
  });
  $('btnStop').onclick = guard(async () => { await api('/api/stop', {}); await refresh(); });
  $('btnExport').onclick = guard(async () => { const r = await api('/api/export', {}); say(t('exported', r.path)); });
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
  $('edRule').onchange = guard(async () => { await api('/api/settings', { ed_rule: $('edRule').value }); await refresh(); });
  $('results').addEventListener('click', guard(async ev => {
    const c = selectedCase();
    const b = ev.target.closest('button');
    if (!c || !b) return;
    if (b.dataset.phase) {
      const body = b.dataset.phase === 'reset' ? { reset: true } : { [b.dataset.phase]: S.frame };
      await api(`/api/cases/${c.id}/phases`, body);
    } else if (b.dataset.exclude) {
      const [phase, chamber, z, on] = b.dataset.exclude.split(',');
      await api(`/api/cases/${c.id}/exclude`, { phase, chamber, slice: +z, exclude: on === '1' });
    } else if (b.dataset.goto) {
      const [f, k] = b.dataset.goto.split(',');
      return gotoFrameSlice(+f, k);
    } else {
      return;
    }
    await refresh();
  }));
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
