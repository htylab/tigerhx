'use strict';

// ---------------------------------------------------------------- strings
const I18N = {
  zh: {
    addFiles: '＋ 檔案 / CSV', addFolder: '＋ 資料夾', model: '模型', stop: '■ 停止',
    runAll: n => `▶ 分割 ${n} 個待處理`, runChecked: n => `▶ 分割已勾選 (${n})`, runNone: '▶ 分割',
    settings: '設定', pathsLabel: '檔案或資料夾路徑（每行一個）',
    include: '只要路徑含有（逗號分隔，空白 = 全部）', exclude: '排除路徑含有', add: '加入', cancel: '取消', close: '關閉',
    saveMat: '輸出舊版格式 .mat', matInput: '.mat 也存一份到原始影像資料夾',
    niiInput: '分割結果 .nii.gz 存到原始影像資料夾', workspace: '工作資料夾', openFolder: '開啟',
    cases: '病例', selectAll: '全選',
    emptyTitle: '開始分割', step1: '加入 cine NIfTI、資料夾，或舊版的 CSV 清單', step2: '按「▶ 分割」，模型會自動下載',
    step3: '逐一確認 Apical 起始 slice，完成後輸出 .mat',
    emptyNote: 'CSV 裡已填好的 Apex 會直接視為已確認。也可以用「開啟結果」查看舊版 GUI 的 .mat。',
    noMatch: '沒有符合篩選的病例。',
    f_all: '全部', f_review: '待確認', f_warn: '有警示', f_pending: '未分割', f_done: '完成', f_error: '錯誤',
    remove: '移除勾選', openCaseFolder: '開啟病例資料夾',
    viewSingle: '單張', viewMosaic: '全部 slice', zoomHeart: '心臟', zoomFull: '整張',
    ovSeg: '分割', ovEdge: '輪廓', ovAha: 'AHA 17', ovNone: '無疊圖',
    opacity: '透明度', hintSelect: '從左側選一個病例', hintEmpty: '加入病例後，影像和分割會顯示在這裡', frame: 'Frame', slice: 'Slice',
    apexStart: 'Apical 起始 slice', setApex: '設為目前 slice', confirmApex: '✓ 確認', confirmNext: '✓ 確認，下一個 →',
    sliceOrder: '順序', baseFirst: 'base → apex', apexFirst: 'apex → base',
    details: '病例資訊', noResults: '選一個病例後，這裡會顯示它的資訊與建議檢查的 slice。',
    st_pending: '未分割', st_queued: '排隊中', st_running: '分割中', st_review: '待確認',
    st_done: '完成', st_error: '錯誤',
    suggested: '自動建議，請確認', confirmed: '已確認',
    lv_basal: 'basal', lv_mid: 'mid', lv_apical: 'apical', lv_apex: 'apex cap',
    added: n => `加入 ${n} 個病例`, queued: n => `${n} 個病例排入佇列`,
    downloading: (m, p) => `下載模型 ${m} … ${p}`, runningCase: (n, p) => `分割中：${n} ${p}`,
    queueLeft: n => `（佇列還有 ${n} 個）`, idle: '待命', apexSaved: k => `Apex 設為 slice ${k}，AHA 與 6/4 分區已重算`,
    allConfirmed: '所有病例的 Apex 都已確認 ✓', nextCase: n => `已確認，下一個：${n}`,
    confirmRemove: n => `從清單移除 ${n} 個病例？（不會刪除檔案）`,
    notSegmented: '尚未分割。按上方「▶ 分割」開始。', error: '錯誤',
    lblLV: 'LV 血池', lblMYO: '心肌', lblRV: 'RV 血池', lblEndo: 'LV 心內膜', lblEpi: 'LV 心外膜',
    ovSectors: '6/4 分區', ovLV: 'LV', ovRV: 'RV', overlay: '疊圖', cmap: '色表',
    openResults: '開啟結果 (.mat)', genCsv: '只產生 CSV', resultsPrompt: '含 *pred*.mat 的資料夾',
    genCsvNote: '不加入病例，只把找到的檔案寫成 CSV（Apex 欄空白），在 Excel 填好後再用「＋ 檔案 / CSV」加入',
    addedFull: (a, u, m) => `加入 ${a} 個病例` + (u ? `，更新 ${u} 個` : '') + (m ? `，${m} 個檔案找不到（見 Log）` : ''),
    genCsvDone: (n, p) => `GenCSV：${n} 個檔案 → ${p}（填好 Apex 後用「＋ 檔案 / CSV」加入）`,
    addTitleFiles: '加入檔案', addTitleFolder: '加入資料夾',
    addNoteBrowser: '貼上 NIfTI、CSV 或資料夾的完整路徑。資料夾會往下搜尋所有子資料夾。',
    addNoteFolder: '會搜尋資料夾裡（含子資料夾）所有 .nii / .nii.gz。',
    filterNote: '例如只要 cine 影像：包含填 cine；排除 mask, _seg 可避開標註檔。',
    secOutputs: '輸出', secFolders: '資料夾', secModels: '模型',
    csvOverride: 'CSV 清單裡每一列的 mat_in_inputdir / nii_in_inputdir 會覆蓋這兩個設定。',
    outputDir: '.mat 輸出資料夾', outputDirHint: '空白 = 工作資料夾/output', csvDir: 'CSV 資料夾',
    downloadAll: '下載全部模型', downloadStarted: n => (n ? `開始下載 ${n} 個模型` : '全部模型都已下載'),
    modelsDownloaded: (a, n) => `已下載 ${a} / ${n} 個模型`,
    log: 'Log', kindMat: '舊結果', matNote: '舊版 GUI 的結果（唯讀）：apex 取自其 6/4 分區，不會寫出任何檔案。',
    csvNote: '來自 CSV：', ownOutputs: (m, n) => `.mat 存原始資料夾 ${m}、.nii.gz 存原始資料夾 ${n}`,
    yes: '是', no: '否', sectorsLegend: (n) => `${n} 分區`,
    wall_anterior: '前壁', wall_anteroseptal: '前間隔', wall_inferoseptal: '下間隔', wall_inferior: '下壁',
    wall_inferolateral: '下側壁', wall_anterolateral: '前側壁', wall_septal: '間隔', wall_lateral: '側壁', wall_apex: '心尖',
    modelMissing: '（首次使用時下載）',
    dtFile: '檔案', dtModel: '模型', dtPhases: 'ED / ES', dtSize: '大小',
    phases: (ed, es) => `frame ${ed} / ${es}`, phasesNote: 'LV 血池最大 / 最小',
    checksTitle: '建議檢查', noChecks: '沒有需要特別看的 slice', show: '查看', flagged: '有警示',
    chkBigger: (z, ch, es, ed) => `slice ${z}：ES 的 ${ch} 比 ED 大（${es} vs ${ed} mm²），可能標到心室外的結構`,
    chkGap: (z, ph, ch, a, n) => `slice ${z}：${ph} 的 ${ch} 疑似漏標（${a} mm²，鄰近 slice ${n} mm²）`,
    chkRing: (z, p) => `slice ${z}：ES 最基底的 LV 只有 ${p}% 被心肌包住，LV 標記可能已進入心房`,
    outputs: '輸出檔案', openOutput: '開啟 .mat 輸出資料夾',
    shortcuts: '快捷鍵', playKey: '播放 / 暫停（空白鍵）', sliceKey: '上一個 / 下一個 slice（↑ ↓）',
    setApexKey: '把目前的 slice 設為 Apical 起始（A）', confirmKey: '確認 Apex，並跳到下一個待確認的病例（Enter）',
    k_frame: '上一個 / 下一個 frame', k_slice: '上一個 / 下一個 slice', k_play: '播放 / 暫停',
    k_apex: '把目前的 slice 設為 Apical 起始', k_confirm: '確認 Apex，跳到下一個待確認的病例',
    k_case: '上一個 / 下一個病例', k_view: '單張 / 全部 slice', k_overlay: '切換疊圖', k_help: '顯示這個說明',
  },
  en: {
    addFiles: '＋ Files / CSV', addFolder: '＋ Folder', model: 'Model', stop: '■ Stop',
    runAll: n => `▶ Segment ${n} pending`, runChecked: n => `▶ Segment checked (${n})`, runNone: '▶ Segment',
    settings: 'Settings', pathsLabel: 'File or folder paths (one per line)',
    include: 'Path contains (comma separated, empty = all)', exclude: 'Skip paths containing', add: 'Add', cancel: 'Cancel', close: 'Close',
    saveMat: 'Write legacy .mat files', matInput: 'Also write the .mat next to the input image',
    niiInput: 'Write the segmentation .nii.gz next to the input image', workspace: 'Workspace', openFolder: 'Open',
    cases: 'Cases', selectAll: 'Select all',
    emptyTitle: 'Get started', step1: 'Add cine NIfTI files, a folder, or a case list (CSV) of the previous GUI',
    step2: 'Press “▶ Segment”; models are downloaded automatically', step3: 'Confirm the first apical slice of each case; the .mat is then written',
    emptyNote: 'Apex values filled in a CSV are taken as confirmed. “Open results” shows .mat files of the previous GUI.',
    noMatch: 'No case matches this filter.',
    f_all: 'All', f_review: 'To confirm', f_warn: 'Flagged', f_pending: 'Not segmented', f_done: 'Done', f_error: 'Error',
    remove: 'Remove checked', openCaseFolder: 'Open case folder',
    viewSingle: 'Single', viewMosaic: 'All slices', zoomHeart: 'Heart', zoomFull: 'Full',
    ovSeg: 'Segmentation', ovEdge: 'Contours', ovAha: 'AHA 17', ovNone: 'No overlay',
    opacity: 'Opacity', hintSelect: 'Select a case on the left', hintEmpty: 'Images and segmentations appear here once cases are added', frame: 'Frame', slice: 'Slice',
    apexStart: 'First apical slice', setApex: 'Use current slice', confirmApex: '✓ Confirm', confirmNext: '✓ Confirm, next →',
    sliceOrder: 'Order', baseFirst: 'base → apex', apexFirst: 'apex → base',
    details: 'Case details', noResults: 'Select a case to see its details and the slices worth a look.',
    st_pending: 'Not segmented', st_queued: 'Queued', st_running: 'Segmenting', st_review: 'To confirm',
    st_done: 'Done', st_error: 'Error',
    suggested: 'suggested, please confirm', confirmed: 'confirmed',
    lv_basal: 'basal', lv_mid: 'mid', lv_apical: 'apical', lv_apex: 'apex cap',
    added: n => `Added ${n} case(s)`, queued: n => `Queued ${n} case(s)`,
    downloading: (m, p) => `Downloading model ${m} … ${p}`, runningCase: (n, p) => `Segmenting: ${n} ${p}`,
    queueLeft: n => ` (${n} more queued)`, idle: 'Idle', apexSaved: k => `Apex set to slice ${k}; AHA and 6/4 sectors updated`,
    allConfirmed: 'The apex of every case is confirmed ✓', nextCase: n => `Confirmed; next: ${n}`,
    confirmRemove: n => `Remove ${n} case(s) from the list? (files are kept)`,
    notSegmented: 'Not segmented yet. Press “▶ Segment” above.', error: 'Error',
    lblLV: 'LV blood', lblMYO: 'Myocardium', lblRV: 'RV blood', lblEndo: 'LV endocardium', lblEpi: 'LV epicardium',
    ovSectors: '6/4 sectors', ovLV: 'LV', ovRV: 'RV', overlay: 'Overlay', cmap: 'Colour map',
    openResults: 'Open results (.mat)', genCsv: 'Only write a CSV', resultsPrompt: 'Folder with *pred*.mat files',
    genCsvNote: 'Adds nothing: writes the files found to a CSV (Apex left blank) to fill in with Excel and add with “＋ Files / CSV”',
    addedFull: (a, u, m) => `Added ${a} case(s)` + (u ? `, updated ${u}` : '') + (m ? `, ${m} file(s) not found (see Log)` : ''),
    genCsvDone: (n, p) => `GenCSV: ${n} file(s) → ${p} (fill in Apex, then add the CSV with “＋ Files / CSV”)`,
    addTitleFiles: 'Add files', addTitleFolder: 'Add folder',
    addNoteBrowser: 'Paste full paths of NIfTI files, CSV case lists or folders. Folders are searched with their subfolders.',
    addNoteFolder: 'Every .nii / .nii.gz in the folder and its subfolders is added.',
    filterNote: 'For example, to take only cine images: contains “cine”; skip “mask, _seg” to leave out label files.',
    secOutputs: 'Outputs', secFolders: 'Folders', secModels: 'Models',
    csvOverride: 'In a CSV case list, the mat_in_inputdir / nii_in_inputdir columns override these two settings per row.',
    outputDir: '.mat output folder', outputDirHint: 'empty = workspace/output', csvDir: 'CSV folder',
    downloadAll: 'Download all models', downloadStarted: n => (n ? `Downloading ${n} model(s)` : 'All models are already downloaded'),
    modelsDownloaded: (a, n) => `${a} of ${n} models downloaded`,
    log: 'Log', kindMat: 'result', matNote: 'Result of the previous GUI (read-only): apex taken from its 6/4 sectors; nothing is written.',
    csvNote: 'From the CSV: ', ownOutputs: (m, n) => `.mat next to input ${m}, .nii.gz next to input ${n}`,
    yes: 'yes', no: 'no', sectorsLegend: (n) => `${n} sectors`,
    wall_anterior: 'Anterior', wall_anteroseptal: 'Anteroseptal', wall_inferoseptal: 'Inferoseptal', wall_inferior: 'Inferior',
    wall_inferolateral: 'Inferolateral', wall_anterolateral: 'Anterolateral', wall_septal: 'Septal', wall_lateral: 'Lateral', wall_apex: 'Apex',
    modelMissing: ' (downloaded on first use)',
    dtFile: 'File', dtModel: 'Model', dtPhases: 'ED / ES', dtSize: 'Size',
    phases: (ed, es) => `frame ${ed} / ${es}`, phasesNote: 'largest / smallest LV blood pool',
    checksTitle: 'Worth checking', noChecks: 'No slice needs a closer look', show: 'Show', flagged: 'flagged',
    chkBigger: (z, ch, es, ed) => `slice ${z}: ${ch} larger at ES than at ED (${es} vs ${ed} mm²), the label may cover a structure outside the ventricle`,
    chkGap: (z, ph, ch, a, n) => `slice ${z}: ${ch} possibly missed at ${ph} (${a} mm², neighbouring slices ${n} mm²)`,
    chkRing: (z, p) => `slice ${z}: only ${p}% of the basal-most LV at ES is enclosed by myocardium, the LV label may include atrium`,
    outputs: 'Output files', openOutput: 'Open the .mat output folder',
    shortcuts: 'Keyboard shortcuts', playKey: 'Play / pause (Space)', sliceKey: 'Previous / next slice (↑ ↓)',
    setApexKey: 'Use the current slice as the first apical slice (A)', confirmKey: 'Confirm the apex and go to the next case to confirm (Enter)',
    k_frame: 'Previous / next frame', k_slice: 'Previous / next slice', k_play: 'Play / pause',
    k_apex: 'Use the current slice as the first apical slice', k_confirm: 'Confirm the apex, go to the next case to confirm',
    k_case: 'Previous / next case', k_view: 'Single / all slices', k_overlay: 'Next overlay', k_help: 'Show this help',
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
  state: null, selected: null, checked: new Set(), filter: 'all',
  view: 'single', zoom: 'heart', overlay: 'seg', opacity: 0.45,
  frame: 0, slice: null, playing: null,
  loadedKey: null, ahaKey: null, loading: false, message: '', error: false,
  cmap: 'gray', logId: 0, logOpen: false,
};
try { S.cmap = localStorage.getItem('tigerhx_cmap') || 'gray'; } catch (e) { /* storage blocked */ }
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const baseName = p => String(p ?? '').split(/[\\/]/).pop();
// long file names may wrap after '_' and '.' instead of anywhere
const wrapName = s => esc(s).replace(/([_.])/g, '$1<wbr>');

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
let toastTimer = null;
function say(msg, isError = false) {
  S.message = msg; S.error = isError;
  const el = $('statusMsg');
  el.textContent = msg;
  el.title = msg;
  el.className = isError ? 'err' : '';
  const toast = $('toast');
  toast.textContent = msg;
  toast.className = `toast${isError ? ' err' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), isError ? 7000 : 3500);
}
const guard = fn => async (...a) => { try { await fn(...a); } catch (e) { say(`${t('error')}: ${e.message}`, true); } };

const selectedCase = () => S.state && S.state.cases.find(c => c.id === S.selected);
const isSegmented = c => !!(c && ['review', 'done'].includes(c.status) && c.levels);
const isFlagged = c => isSegmented(c) && (c.checks || []).length > 0;

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
  document.querySelectorAll('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); });
  $('btnLang').textContent = lang === 'zh' ? 'EN' : '中文';
  $('btnSetApex').textContent = t('setApex');
  const keys = [['← →', 'k_frame'], ['↑ ↓', 'k_slice'], ['Space', 'k_play'], ['A', 'k_apex'], ['Enter', 'k_confirm'],
    ['N  P', 'k_case'], ['M', 'k_view'], ['O', 'k_overlay'], ['?', 'k_help']];
  $('keysTable').innerHTML = keys.map(([k, d]) =>
    `<tr><td>${k.split(/\s+/).map(x => `<kbd>${esc(x)}</kbd>`).join(' ')}</td><td>${esc(t(d))}</td></tr>`).join('');
}

function render() {
  const st = S.state;
  if (!st) return;
  if (!selectedCase() && st.cases.length) {
    const first = st.cases.find(c => c.status === 'review') || st.cases[0];
    S.selected = first.id;
  }
  renderTop(st);
  renderCases(st);
  renderStatus(st);
  const c = selectedCase();
  renderViewerBars(c);
  renderResults(c);
  if (c) maybeLoadCase(c);
  else { $('canvasHint').textContent = t(st.cases.length ? 'hintSelect' : 'hintEmpty'); $('canvasHint').classList.remove('hidden'); }
  document.querySelectorAll('.viewer-bar').forEach(el => el.classList.toggle('hidden', !c));
  $('apexBar').classList.toggle('hidden', !isSegmented(c));
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
  const sep = st.workspace.includes('\\') ? '\\' : '/';
  $('csvDirText').textContent = `${st.workspace}${sep}csv`;
  $('modelSummary').textContent = t('modelsDownloaded', st.models.filter(m => m.downloaded).length, st.models.length);
  if (document.activeElement !== $('outputDir')) $('outputDir').value = st.settings.output_dir || '';
  $('outputDir').placeholder = `${st.output_dir}  (${t('outputDirHint')})`;
  $('outputDir').disabled = !st.settings.save_mat;
  $('btnOpenOutput').disabled = !st.settings.save_mat;

  const checked = [...S.checked].filter(id => st.cases.some(c => c.id === id));
  const pending = st.cases.filter(c => (c.status === 'pending' || c.status === 'error') && c.kind !== 'mat').length;
  const run = $('btnRun');
  run.textContent = checked.length ? t('runChecked', checked.length) : pending ? t('runAll', pending) : t('runNone');
  run.disabled = checked.length ? false : pending === 0;
  const busy = !!(st.running || st.queued);
  $('btnStop').classList.toggle('hidden', !busy);

  // progress of the running case or model download, in the status bar
  const ms = st.model_status;
  const running = st.running && st.cases.find(x => x.id === st.running);
  const frac = ms.state === 'downloading' ? ms.progress : running ? running.progress : null;
  $('topProgress').classList.toggle('hidden', frac === null);
  if (frac !== null) $('topProgressBar').style.width = `${(frac * 100).toFixed(1)}%`;
}

function pill(c) {
  const pct = c.status === 'running' ? ` ${Math.round(c.progress * 100)}%` : '';
  return `<span class="pill ${c.status}"><span class="dot"></span>${esc(t('st_' + c.status))}${pct}</span>`;
}

const FILTERS = {
  all: () => true,
  review: c => c.status === 'review',
  warn: isFlagged,
  pending: c => ['pending', 'queued', 'running'].includes(c.status),
  done: c => c.status === 'done',
  error: c => c.status === 'error',
};

function renderFilters(st) {
  const box = $('caseFilters');
  const counts = Object.fromEntries(Object.entries(FILTERS).map(([k, f]) => [k, st.cases.filter(f).length]));
  if (counts[S.filter] === 0 && S.filter !== 'all') S.filter = 'all';
  // only filters that would narrow the list are shown
  const shown = st.cases.length ? Object.keys(FILTERS).filter(k => k === 'all' || (counts[k] && counts[k] < counts.all)) : [];
  const html = shown.length > 1 ? shown.map(k =>
    `<button data-filter="${k}" class="${k === S.filter ? 'on' : ''}">${esc(t('f_' + k))}<b>${counts[k]}</b></button>`).join('') : '';
  if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
}

function visibleCases(st) {
  return st.cases.filter(FILTERS[S.filter] || FILTERS.all);
}

function caseRow(c) {
  const name = c.name.split('/');
  const leaf = name.pop();
  const apex = c.apex_slice === null || c.apex_slice === undefined ? '' :
    `<span title="${esc(t('apexStart'))}">Apex ${esc(c.apex_slice)}${c.apex_confirmed ? '' : '?'}</span>`;
  const nWarn = (c.checks || []).length;
  const qc = !isSegmented(c) ? '' : nWarn
    ? `<span class="warn-mark" title="${esc(t('checksTitle'))}: ${nWarn}">⚠ ${nWarn}</span>` : '<span class="ok-mark">✓</span>';
  const bar = c.status === 'running' ? `<span class="bar"><i style="width:${(c.progress * 100).toFixed(1)}%"></i></span>` : '';
  const msg = c.message && c.status === 'error' ? `<div class="case-msg">${esc(c.message)}</div>` : '';
  const folder = name.length ? `<bdi>${esc(name.join('/'))}/</bdi>` : '';
  return `<li class="row${c.id === S.selected ? ' selected' : ''}" data-id="${c.id}" title="${esc(c.path)}">
    <input type="checkbox" data-check="${c.id}"${S.checked.has(c.id) ? ' checked' : ''} aria-label="${esc(c.name)}">
    <span class="case-name">${wrapName(leaf)}</span>${pill(c)}
    <span></span><div class="case-sub"><span class="folder">${folder}</span><span class="meta">${
      c.kind === 'mat' ? `<span class="tag mat">${esc(t('kindMat'))}</span>` : ''}${bar}${apex}${qc}</span></div>${msg}</li>`;
}

function renderCases(st) {
  const body = $('caseBody');
  $('caseCount').textContent = st.cases.length ? `${st.cases.length}` : '';
  $('caseEmpty').classList.toggle('hidden', st.cases.length > 0);
  renderFilters(st);
  const vis = visibleCases(st);
  $('caseNoMatch').classList.toggle('hidden', !st.cases.length || vis.length > 0);
  const rows = vis.map(caseRow).join('');
  if (body.dataset.html !== rows) { body.innerHTML = rows; body.dataset.html = rows; }
  $('checkAll').checked = vis.length > 0 && vis.every(c => S.checked.has(c.id));
  $('checkAll').disabled = !vis.length;
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

const apexSide = (c, k) => (c.base_first === false ? k <= c.apex_slice : k >= c.apex_slice);

function renderLadder(c, seg, nz) {
  const box = $('ladder');
  const flags = new Set(seg ? (c.checks || []).map(k => k.slice) : []);
  const key = JSON.stringify([c && c.id, nz, seg && c.levels, seg && c.apex_slice, seg && c.base_first,
    seg && c.apex_confirmed, [...flags], S.slice, !!S.loadedKey, lang]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.classList.toggle('apex-confirmed', !!(seg && c.apex_confirmed));
  box.classList.toggle('apex-first', !!(seg && c.base_first === false));
  const parts = [];
  for (let k = 0; k < nz; k++) {
    const level = seg && c.levels ? c.levels[k] : null;
    const cls = [level ? `l-${level === 'apex' ? 'apex' : level}` : '', k === S.slice ? 'cur' : '',
      seg && c.apex_slice === k ? 'apex-start' : '', flags.has(k) ? 'flag' : ''].filter(Boolean).join(' ');
    const tip = [`slice ${k}`, level ? t('lv_' + level) : '', seg && c.apex_slice === k ? t('apexStart') : '',
      flags.has(k) ? `⚠ ${t('flagged')}` : ''].filter(Boolean).join(' · ');
    parts.push(`<button data-slice="${k}" class="${cls}" title="${esc(tip)}"${S.loadedKey ? '' : ' disabled'}>${k}</button>`);
  }
  box.innerHTML = parts.join('');
  const cur = box.querySelector('.cur');
  if (cur) cur.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function nextToConfirm(c) {
  const list = S.state.cases;
  const i = list.findIndex(x => x.id === c.id);
  for (let n = 1; n < list.length; n++) {
    const x = list[(i + n) % list.length];
    if (x.status === 'review') return x;
  }
  return null;
}

function renderViewerBars(c) {
  const seg = isSegmented(c);
  const nt = c && c.shape ? c.shape[3] : 1;
  const nz = c && c.shape ? c.shape[2] : 0;
  const fr = $('frame');
  fr.max = Math.max(0, nt - 1);
  if (+fr.value !== S.frame) fr.value = S.frame;
  const phase = seg && nt > 1 ? (S.frame === c.ed ? ' ED' : S.frame === c.es ? ' ES' : '') : '';
  $('frameText').textContent = c && c.shape ? `${S.frame}/${nt - 1}${phase}` : '–';
  $('btnED').disabled = !seg || nt < 2;
  $('btnES').disabled = !seg || nt < 2;
  $('btnPlay').disabled = !S.loadedKey || nt < 2;
  $('sliceText').textContent = S.slice === null || !nz ? '–' : `${S.slice}/${nz - 1}`;
  $('btnSlicePrev').disabled = !S.loadedKey;
  $('btnSliceNext').disabled = !S.loadedKey;
  fr.disabled = !S.loadedKey || nt < 2;
  renderLadder(c, seg, nz);

  $('apexValue').textContent = seg && c.apex_slice !== null ? c.apex_slice : '–';
  const tag = $('apexTag');
  const showTag = seg && c.apex_slice !== null && c.kind !== 'mat';
  tag.className = `tag ${seg && c.apex_confirmed ? 'ok' : 'warn'}${showTag ? '' : ' hidden'}`;
  tag.textContent = seg && c.apex_confirmed ? `✓ ${t('confirmed')}` : t('suggested');
  const readOnly = !seg || c.kind === 'mat';
  $('apexBar').classList.toggle('needs', seg && !c.apex_confirmed && c.kind !== 'mat');
  $('btnSetApex').disabled = readOnly || S.slice === null || S.slice === c.apex_slice;
  const confirm = $('btnConfirmApex');
  const next = seg ? nextToConfirm(c) : null;
  confirm.textContent = next && !c.apex_confirmed ? t('confirmNext') : t('confirmApex');
  confirm.disabled = readOnly || c.apex_confirmed || c.apex_slice === null;
  const so = $('sliceOrder');
  so.disabled = readOnly;
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

// ---------------------------------------------------------------- details
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
    c.kind, c.mat_in_inputdir, c.nii_in_inputdir, c.apex_confirmed, c.shape, Math.round((c.progress || 0) * 100)]) : `none${lang}`;
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  if (!c) { box.innerHTML = `<div class="empty">${esc(t('noResults'))}</div>`; return; }
  const yn = v => (v === null || v === undefined ? '–' : t(v ? 'yes' : 'no'));
  const own = c.mat_in_inputdir !== null || c.nii_in_inputdir !== null;
  const rows = [[t('dtFile'), `<span title="${esc(c.path)}">${wrapName(baseName(c.path))}</span>`]];
  if (c.shape) rows.push([t('dtSize'), esc(c.shape.join(' × '))]);
  if (c.model) rows.push([t('dtModel'), esc(c.model)]);
  if (isSegmented(c) && c.shape[3] > 1) rows.push([t('dtPhases'), `<span title="${esc(t('phasesNote'))}">${esc(t('phases', c.ed, c.es))}</span>`]);
  const head = `<div class="case-title"><span>${wrapName(baseName(c.name))}</span> ${pill(c)}</div>
    <dl class="meta">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>` +
    (c.kind === 'mat' ? `<div class="note">${esc(t('matNote'))}</div>` : '') +
    (own ? `<div class="note">${esc(t('csvNote') + t('ownOutputs', yn(c.mat_in_inputdir), yn(c.nii_in_inputdir)))}</div>` : '');
  if (!isSegmented(c)) {
    const msg = c.status === 'error' ? `<div class="case-msg">${esc(c.message)}</div>` : '';
    const hint = ['pending', 'error'].includes(c.status) ? `<div class="empty">${esc(t('notSegmented'))}</div>` : '';
    box.innerHTML = `${head}${hint}${msg}`;
    return;
  }
  const checks = (c.checks || []).map(k => `<li><span class="ico">⚠</span>
      <span>${esc(checkText(k))}</span>
      <button class="small" data-goto="${frameOf(c, k)},${k.slice}">${esc(t('show'))}</button></li>`).join('');
  const outputs = Object.values(c.outputs || {}).map(p => `<li title="${esc(p)}">${wrapName(baseName(p))}</li>`).join('');
  box.innerHTML = `${head}
    ${checks ? `<div class="card"><div class="card-title">⚠ ${esc(t('checksTitle'))}</div><ul class="checks">${checks}</ul></div>`
    : `<div class="card ok"><div class="card-title">✓ ${esc(t('noChecks'))}</div></div>`}
    ${outputs ? `<div class="card"><div class="card-title">${esc(t('outputs'))}</div><ul class="outputs">${outputs}</ul>
      <div><button class="small" data-open="output">${esc(t('openOutput'))}</button></div></div>` : ''}`;
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
  isRadiologicalConvention: false, crosshairWidth: 0, crosshairGap: 8,
});
nv.attachToCanvas($('gl'));
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

// All-slices view: each tile gets its slice number and level, the current
// slice a frame and the first apical slice a marker.
function drawTileLabels(c) {
  const ctx = ovCtx;
  const nz = nv.volumes[0].hdr.dims[3];
  const dpr = nv.uiData.dpr || 1;
  const seg = isSegmented(c);
  ctx.save();
  ctx.font = `600 ${11 * dpr}px system-ui, sans-serif`;
  ctx.textBaseline = 'top';
  for (const tile of nv.screenSlices) {
    if (tile.axCorSag !== 0 || !tile.leftTopWidthHeight) continue;
    const k = Math.round(tile.sliceFrac * nz - 0.5);
    const [L, T, W, H] = tile.leftTopWidthHeight;
    const level = seg && c.levels ? c.levels[k] : '';
    const apex = seg && c.apex_slice === k;
    const text = `${k}${level ? ` · ${t('lv_' + level)}` : ''}${apex ? ' · apex' : ''}`;
    const pad = 4 * dpr;
    const w = ctx.measureText(text).width + 2 * pad;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(L + 2 * dpr, T + 2 * dpr, w, 16 * dpr);
    ctx.fillStyle = apex ? (c.apex_confirmed ? '#6ee06e' : '#ffc94d') : '#f0efec';
    ctx.fillText(text, L + 2 * dpr + pad, T + 4 * dpr);
    if (k === S.slice) {
      ctx.lineWidth = 2 * dpr;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.strokeRect(L + dpr, T + dpr, W - 2 * dpr, H - 2 * dpr);
    }
  }
  ctx.restore();
}

function drawOverlay() {
  drawShapes();
  if (S.loadedKey && S.view === 'mosaic' && nv.volumes.length) drawTileLabels(selectedCase());
}

function drawShapes() {
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
  if (r.added && S.filter !== 'all') S.filter = 'all';
  await refresh();
}
// native file dialogs: the Rust app's bridge or the pywebview API
const native = () => window.tigerhxNative || (window.pywebview && window.pywebview.api);

function showAddDialog(paths = [], folder = false) {
  $('addTitle').textContent = t(folder ? 'addTitleFolder' : 'addTitleFiles');
  $('addNote').textContent = t(paths.length ? 'addNoteFolder' : 'addNoteBrowser');
  $('addPaths').value = paths.join('\n');
  $('addDialog').showModal();
  (paths.length ? $('btnAddConfirm') : $('addPaths')).focus();
}
const dialogPaths = () => $('addPaths').value.split('\n').map(s => s.trim()).filter(Boolean);

function selectCase(id) {
  if (!id || id === S.selected) return;
  S.selected = id;
  $('caseBody').dataset.html = '';
  render();
  const row = document.querySelector(`#caseBody li[data-id="${id}"]`);
  if (row) row.scrollIntoView({ block: 'nearest' });
}
function stepCase(d) {
  const list = visibleCases(S.state);
  if (!list.length) return;
  const i = list.findIndex(c => c.id === S.selected);
  selectCase(list[Math.max(0, Math.min(list.length - 1, (i < 0 ? 0 : i + d)))].id);
}

const postApex = async (apex, baseFirst) => {
  const c = selectedCase();
  await api(`/api/cases/${c.id}/apex`, { apex_slice: apex, base_first: baseFirst });
  say(t('apexSaved', apex));
  await refresh();
};
const setApexHere = guard(async () => {
  const c = selectedCase();
  if (!isSegmented(c) || c.kind === 'mat' || S.slice === null) return;
  await postApex(S.slice, null);
});
const confirmApex = guard(async () => {
  const c = selectedCase();
  if (!isSegmented(c) || c.kind === 'mat' || c.apex_confirmed || c.apex_slice === null) return;
  await api(`/api/cases/${c.id}/apex`, { apex_slice: c.apex_slice, base_first: null });
  await refresh();
  const next = nextToConfirm(selectedCase());
  if (next) { say(t('nextCase', next.name)); selectCase(next.id); } else say(t('allConfirmed'));
});
const OVERLAYS = ['seg', 'edge', 'aha', 'sectors', 'lv', 'rv', 'none'];
function setOverlay(mode) {
  S.overlay = mode;
  applyOverlay();
  renderViewerBars(selectedCase());
}
function setView(v) {
  S.view = v;
  applyView();
  renderViewerBars(selectedCase());
  drawOverlay();
}

function toggleLog(open = !S.logOpen) {
  S.logOpen = open;
  $('logPanel').classList.toggle('hidden', !S.logOpen);
  $('btnLog').classList.remove('alert');
  if (S.logOpen) $('logBody').scrollTop = $('logBody').scrollHeight;
}

function bind() {
  const addFiles = guard(async () => {
    if (!native()) return showAddDialog();
    const files = await native().pick_files();
    if (files && files.length) await addPaths(files);
  });
  const addFolder = guard(async () => {
    if (!native()) return showAddDialog([], true);
    const dirs = await native().pick_folder();
    if (dirs && dirs.length) showAddDialog(dirs, true);
  });
  const openResults = guard(async () => {
    let dirs = null;
    if (native()) dirs = await native().pick_folder();
    else {
      const d = window.prompt(t('resultsPrompt'), S.state ? S.state.output_dir : '');
      dirs = d ? [d] : null;
    }
    if (dirs && dirs.length) await addPaths(dirs, '', '', true);
  });
  $('btnAddFiles').onclick = addFiles;
  $('btnEmptyFiles').onclick = addFiles;
  $('btnAddFolder').onclick = addFolder;
  $('btnEmptyFolder').onclick = addFolder;
  $('btnOpenResults').onclick = openResults;
  $('btnEmptyResults').onclick = openResults;
  $('btnAddConfirm').onclick = guard(async () => {
    const paths = dialogPaths();
    if (!paths.length) return $('addPaths').focus();
    await addPaths(paths, $('addInclude').value, $('addExclude').value);
    $('addDialog').close();
  });
  $('btnAddCancel').onclick = () => $('addDialog').close();
  $('btnGenCsv').onclick = guard(async () => {
    const paths = dialogPaths();
    if (!paths.length) return $('addPaths').focus();
    const r = await api('/api/gencsv', { paths, include: $('addInclude').value, exclude: $('addExclude').value });
    say(t('genCsvDone', r.count, r.path));
    $('addDialog').close();
  });

  $('btnSettings').onclick = () => $('settingsDialog').showModal();
  $('btnHelp').onclick = () => $('helpDialog').showModal();
  $('outputDir').onchange = guard(async () => {
    await api('/api/settings', { output_dir: $('outputDir').value });
    await refresh();
  });
  $('btnOpenOutput').onclick = guard(() => api('/api/open', { folder: 'output' }));
  $('btnOpenCsv').onclick = guard(() => api('/api/open', { folder: 'csv' }));
  $('btnOpenWs').onclick = guard(() => api('/api/open', {}));
  $('btnDownloadAll').onclick = guard(async () => {
    const r = await api('/api/models/download', {});
    say(t('downloadStarted', r.downloading));
    await refresh();
  });
  document.querySelectorAll('[data-setting]').forEach(el => {
    el.onchange = guard(async () => { await api('/api/settings', { [el.dataset.setting]: el.checked }); await refresh(); });
  });
  $('btnLog').onclick = () => toggleLog();
  $('btnLogClose').onclick = () => toggleLog(false);
  $('model').onchange = guard(async () => { await api('/api/settings', { model: $('model').value }); await refresh(); });
  $('btnRun').onclick = guard(async () => {
    const ids = [...S.checked].filter(id => S.state.cases.some(c => c.id === id));
    const r = await api('/api/run', { ids });
    say(t('queued', r.queued));
    S.checked.clear();
    await refresh();
  });
  $('btnStop').onclick = guard(async () => { await api('/api/stop', {}); await refresh(); });
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
    ['caseBody', 'caseFilters'].forEach(id => { $(id).dataset.html = ''; });
    render();
    drawOverlay();
  };

  $('caseFilters').addEventListener('click', ev => {
    const b = ev.target.closest('button[data-filter]');
    if (!b) return;
    S.filter = b.dataset.filter;
    $('caseFilters').dataset.html = '';
    render();
  });
  $('caseBody').addEventListener('click', ev => {
    const cb = ev.target.closest('[data-check]');
    if (cb) {
      if (cb.checked) S.checked.add(cb.dataset.check); else S.checked.delete(cb.dataset.check);
      $('caseBody').dataset.html = '';
      return render();
    }
    const row = ev.target.closest('li.row');
    if (row) selectCase(row.dataset.id);
  });
  $('checkAll').onchange = () => {
    const vis = visibleCases(S.state).map(c => c.id);
    if ($('checkAll').checked) vis.forEach(id => S.checked.add(id)); else vis.forEach(id => S.checked.delete(id));
    $('caseBody').dataset.html = '';
    render();
  };

  document.querySelectorAll('#viewMode button').forEach(b => { b.onclick = () => setView(b.dataset.v); });
  document.querySelectorAll('#zoomMode button').forEach(b => {
    b.onclick = () => { S.zoom = b.dataset.v; renderViewerBars(selectedCase()); render(); };
  });
  $('overlayMode').onchange = () => setOverlay($('overlayMode').value);
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
  $('ladder').addEventListener('click', ev => {
    const b = ev.target.closest('button[data-slice]');
    if (b) goToSlice(+b.dataset.slice);
  });
  $('btnSetApex').onclick = setApexHere;
  $('btnConfirmApex').onclick = confirmApex;
  $('sliceOrder').onchange = guard(() => postApex(selectedCase().apex_slice, $('sliceOrder').value === '1'));
  $('results').addEventListener('click', ev => {
    const o = ev.target.closest('button[data-open]');
    if (o) return guard(() => api('/api/open', { folder: o.dataset.open }))();
    const b = ev.target.closest('button[data-goto]');
    if (!b || !selectedCase()) return;
    const [f, k] = b.dataset.goto.split(',');
    gotoFrameSlice(+f, k);
  });

  document.addEventListener('keydown', ev => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.target.closest('input, textarea, select, dialog')) return;
    if (ev.key === 'Escape' && S.logOpen) return toggleLog(false);
    if (ev.key === '?') { ev.preventDefault(); return $('helpDialog').showModal(); }
    const k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    const keys = {
      n: () => stepCase(1), p: () => stepCase(-1),
    };
    if (S.loadedKey) {
      Object.assign(keys, {
        ArrowLeft: () => setFrame(S.frame - 1), ArrowRight: () => setFrame(S.frame + 1),
        ArrowUp: () => goToSlice(S.slice - 1), ArrowDown: () => goToSlice(S.slice + 1), ' ': togglePlay,
        a: setApexHere, Enter: confirmApex,
        m: () => setView(S.view === 'mosaic' ? 'single' : 'mosaic'),
        o: () => { if (isSegmented(selectedCase())) setOverlay(OVERLAYS[(OVERLAYS.indexOf(S.overlay) + 1) % OVERLAYS.length]); },
      });
    }
    // a focused button keeps Enter and Space
    if ((k === 'Enter' || k === ' ') && ev.target.closest('button')) return;
    if (keys[k]) { ev.preventDefault(); keys[k](); }
  });
  window.addEventListener('resize', () => { if (S.view === 'mosaic') applyView(); });
}

applyI18n();
bind();
poll();
