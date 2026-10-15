"""End-to-end test of a built TigerHx GUI executable in its own window, on
Windows (CI, after the build).

Starts the executable with its native window, connects to the page through
the WebView2 debugging port, clicks through every way of adding cases (with
the real Windows file and folder dialogs, answered through pywinauto),
segments, confirms, changes settings and closes the window.  After every
step the window must still respond.

    pip install playwright pywinauto pillow numpy nibabel scipy onnx
    python tools/ui_test_exe.py dist/tigerhxgui.exe [screenshot folder]
"""
from __future__ import annotations

import asyncio
import ctypes
import http.cookiejar
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
from phantom import phantom, phantom_image, tiny_model  # noqa: E402

MODEL = 'cine4d_v9999_xy_tiny.onnx'
PORT = 9333


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- test data
def make_data(root):
    """Inputs for every way in: loose files, a folder tree, a case list."""
    img = phantom_image(phantom()).astype(np.int16)
    aff = np.diag([1.5, 1.5, 8, 1])
    d = {k: os.path.join(root, k) for k in ('files', 'tree', 'csv', 'models')}
    for p in d.values():
        os.makedirs(p)
    for name in ('a1_cine', 'a2_cine', 'a3_cine'):
        nib.save(nib.Nifti1Image(img, aff), os.path.join(d['files'], f'{name}.nii.gz'))
    for sub in ('s1', 's2'):
        os.makedirs(os.path.join(d['tree'], sub))
        nib.save(nib.Nifti1Image(img, aff), os.path.join(d['tree'], sub, f'{sub}_cine.nii.gz'))
        nib.save(nib.Nifti1Image(img, aff), os.path.join(d['tree'], sub, f'{sub}_mask.nii.gz'))
    nib.save(nib.Nifti1Image(img, aff), os.path.join(d['csv'], 'c1.nii.gz'))
    with open(os.path.join(d['csv'], 'list.csv'), 'w', encoding='utf-8') as f:
        f.write('Filename,Apex,mat_in_inputdir,nii_in_inputdir\nc1.nii.gz,4,,\nnot_there.nii.gz,3,,\n')
    tiny_model(os.path.join(d['models'], MODEL))
    return d


# ---------------------------------------------------------------- the app
class Gui:
    def __init__(self, exe, root, models):
        env = dict(os.environ, TIGERHX_MODEL_DIR=models, TIGERHX_WEBVIEW_DEBUG_PORT=str(PORT))
        self.ws = os.path.join(root, 'ws')
        self.proc = subprocess.Popen([exe, '-w', self.ws], env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, errors='replace')
        self.lines = []
        self.url = None
        threading.Thread(target=self._read, daemon=True).start()
        for _ in range(600):
            if self.url or self.proc.poll() is not None:
                break
            time.sleep(0.1)
        assert self.url, f'no URL printed, exit code {self.proc.poll()}'
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.opener.open(self.url).read()
        self.base = self.url.split('/?')[0]

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip())
            print('|', line.rstrip(), flush=True)
            if 'GUI:' in line and not self.url:
                self.url = line.split('GUI:', 1)[1].strip()

    def api(self, path, body=None, timeout=15):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data, {'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=timeout) as r:
            return json.loads(r.read())

    def cases(self):
        return self.api('/api/state')['cases']

    def pids(self):
        # a one-file PyInstaller build runs the program in a child process
        import psutil
        try:
            return {self.proc.pid} | {c.pid for c in psutil.Process(self.proc.pid).children(recursive=True)}
        except psutil.NoSuchProcess:
            return set()

    def windows(self):
        from pywinauto import Desktop
        return [w for pid in self.pids()
                for w in Desktop(backend='win32').windows(process=pid, visible_only=True)]

    def main_window(self):
        for w in self.windows():
            if w.class_name() != '#32770' and 'TigerHx' in w.window_text():
                return w
        return None

    def hung(self):
        w = self.main_window()
        return bool(w and ctypes.windll.user32.IsHungAppWindow(w.handle))


async def check_alive(gui, page, what):
    """The process runs, the window's thread answers, the page runs script
    and the server answers."""
    assert gui.proc.poll() is None, f'{what}: the program exited with {gui.proc.poll()}'
    for _ in range(20):
        if not gui.hung():
            break
        await asyncio.sleep(0.5)
    else:
        raise AssertionError(f'{what}: the window does not respond (IsHungAppWindow)')
    try:
        await asyncio.wait_for(page.evaluate('1 + 1'), 10)
    except asyncio.TimeoutError:
        raise AssertionError(f'{what}: the page does not run script')
    await asyncio.to_thread(gui.api, '/api/state')


# ---------------------------------------------------------------- native dialogs
def dialogs(gui):
    return [w for w in gui.windows() if w.class_name() == '#32770']


def find_dialog(gui, title, ignore=(), timeout=20):
    """A new, enabled dialog with this title (not one still closing)."""
    end = time.time() + timeout
    while True:
        for w in dialogs(gui):
            try:
                if w.handle not in ignore and w.window_text() == title and w.is_enabled():
                    return w
            except Exception:  # closed meanwhile
                pass
        if time.time() >= end:
            return None
        time.sleep(0.25)


def wait_closed(gui, timeout=15):
    end = time.time() + timeout
    while dialogs(gui):
        if time.time() >= end:
            return False
        time.sleep(0.25)
    return True


def describe(gui):
    """Every window of the program, for a failure message."""
    out = []
    for w in gui.windows():
        try:
            out.append(f'{w.class_name()!r} {w.window_text()!r} enabled={w.is_enabled()} '
                       f'hung={bool(ctypes.windll.user32.IsHungAppWindow(w.handle))}')
        except Exception as e:
            out.append(f'? {e}')
    return '\n  '.join(out)


def spec(w):
    from pywinauto import Desktop
    return Desktop(backend='win32').window(handle=w.handle)


def button(dlg, control_id):
    for b in dlg.descendants(class_name='Button'):
        if b.control_id() == control_id and b.is_visible():
            return b
    dlg.print_control_identifiers()
    raise AssertionError(f'no button {control_id} in the dialog')


def name_box(dlg):
    """The File name / Folder box of a common item dialog."""
    edits = [e for e in dlg.descendants(class_name='Edit') if e.is_visible()]
    for e in edits:
        if e.parent().class_name() in ('ComboBox', 'ComboBoxEx32'):
            return e
    if edits:
        return edits[-1]
    dlg.print_control_identifiers()
    raise AssertionError('no file name box in the dialog')


def answer(dlg, steps):
    """steps: None to cancel, else texts typed into the name box, each
    confirmed with the OK button (a folder first navigates there)."""
    spec(dlg).wait('ready', timeout=10)
    time.sleep(0.5)
    if steps is None:
        button(dlg, 2).click()
        return
    for text in steps:
        box = name_box(dlg)
        box.set_edit_text(text)
        time.sleep(0.3)
        button(dlg, 1).click()
        time.sleep(0.8)


OPEN, FOLDER = 'Open', 'Select Folder'  # the dialogs' titles on English Windows


async def pick(gui, page, selector, title, steps, what):
    log('--', what)
    assert await asyncio.to_thread(wait_closed, gui), f'{what}: a dialog is still open:\n  {describe(gui)}'
    before = {w.handle for w in dialogs(gui)}
    await page.click(selector, no_wait_after=True)
    dlg = await asyncio.to_thread(find_dialog, gui, title, before)
    assert dlg is not None, f'{what}: no {title!r} dialog appeared; windows:\n  {describe(gui)}'
    try:
        await asyncio.to_thread(answer, dlg, steps)
    except Exception as e:
        raise AssertionError(f'{what}: {e!r}; windows:\n  {describe(gui)}')
    assert await asyncio.to_thread(wait_closed, gui), f'{what}: the dialog did not close:\n  {describe(gui)}'
    await asyncio.sleep(0.5)
    await check_alive(gui, page, what)


async def wait_for(cond, what, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        v = await asyncio.to_thread(cond)
        if v:
            return v
        await asyncio.sleep(0.4)
    raise AssertionError(f'timed out: {what}')


# ---------------------------------------------------------------- scenarios
async def scenarios(gui, page, d, shots):
    async def shot(name):
        if shots:
            await page.screenshot(path=os.path.join(shots, f'{name}.png'))

    n = lambda: len(gui.cases())  # noqa: E731
    page.on('dialog', lambda dl: asyncio.ensure_future(dl.accept()))
    await page.wait_for_function("typeof S !== 'undefined' && S.state", timeout=30000)
    await check_alive(gui, page, 'start')
    await shot('00_start')

    await pick(gui, page, '#btnAddFiles', OPEN, None, 'files: cancel')
    assert n() == 0
    await pick(gui, page, '#btnAddFiles', OPEN, [os.path.join(d['files'], 'a1_cine.nii.gz')], 'files: one NIfTI')
    await wait_for(lambda: n() == 1, 'one case added')
    await pick(gui, page, '#btnAddFiles', OPEN, [d['files'], '"a2_cine.nii.gz" "a3_cine.nii.gz"'], 'files: two NIfTI')
    await wait_for(lambda: n() == 3, 'two more cases')
    await pick(gui, page, '#btnAddFiles', OPEN, [os.path.join(d['csv'], 'list.csv')], 'files: case list CSV')
    await wait_for(lambda: n() == 4, 'the CSV case')
    c1 = next(c for c in gui.cases() if c['name'] == 'c1')
    assert c1['apex_slice'] == 4 and c1['apex_confirmed'], c1
    await shot('01_files')

    await pick(gui, page, '#btnAddFolder', FOLDER, None, 'folder: cancel')
    assert await page.evaluate("!document.getElementById('addDialog').open")
    await pick(gui, page, '#btnAddFolder', FOLDER, [d['tree']], 'folder: choose')
    await page.wait_for_function("document.getElementById('addDialog').open", timeout=10000)
    await shot('02_folder_dialog')
    await page.fill('#addInclude', 'cine')
    await page.fill('#addExclude', 'mask')
    await page.click('#btnAddConfirm')
    await wait_for(lambda: n() == 6, 'folder cases')
    await check_alive(gui, page, 'folder: add')

    await pick(gui, page, '#btnAddFolder', FOLDER, [d['tree']], 'folder: GenCSV')
    await page.wait_for_function("document.getElementById('addDialog').open", timeout=10000)
    await page.click('#btnGenCsv')
    await wait_for(lambda: os.path.isdir(os.path.join(gui.ws, 'csv')) and os.listdir(os.path.join(gui.ws, 'csv')),
                   'the GenCSV file')
    assert n() == 6

    for k in range(4):
        await pick(gui, page, '#btnAddFiles' if k % 2 else '#btnAddFolder', OPEN if k % 2 else FOLDER, None,
                   f'open and cancel #{k + 1}')

    log('-- segment')
    gui.api('/api/settings', {'model': MODEL, 'save_mat': True})
    await page.click('#btnRun')
    await wait_for(lambda: all(c['status'] in ('review', 'done') for c in gui.cases()), 'segmentation', 300)
    await check_alive(gui, page, 'segment')
    await shot('03_segmented')

    log('-- confirm with Enter')
    review = [c for c in gui.cases() if c['status'] == 'review']
    assert review, 'nothing to confirm'
    await page.evaluate(f"selectCase('{review[0]['id']}')")
    await page.wait_for_function("document.getElementById('canvasHint').classList.contains('hidden')", timeout=30000)
    await page.locator('body').press('Enter')
    await wait_for(lambda: next(c for c in gui.cases() if c['id'] == review[0]['id'])['status'] == 'done', 'confirm')
    await page.click('#ladder button[data-slice="2"]')
    for key in ('m', 'o', 'm', 'ArrowDown', 'ArrowRight', 'a'):
        await page.locator('body').press(key)
    await check_alive(gui, page, 'viewer keys')
    await shot('04_review')

    log('-- settings, help, log, language')
    await page.click('#btnSettings')
    await page.click('[data-setting="nii_in_inputdir"]')
    await page.click('[data-setting="nii_in_inputdir"]')
    await page.keyboard.press('Escape')
    await page.click('#btnHelp')
    await page.keyboard.press('Escape')
    await page.click('#btnLog')
    await page.click('#btnLogClose')
    await page.click('#btnLang')
    await page.click('#btnLang')
    await check_alive(gui, page, 'settings')

    out = gui.api('/api/state')['output_dir']
    await pick(gui, page, '#btnOpenResults', FOLDER, [out], 'open results folder')
    await wait_for(lambda: any(c['kind'] == 'mat' for c in gui.cases()), 'opened results')
    await wait_for(lambda: all(c['status'] in ('review', 'done') for c in gui.cases()), 'results read', 120)
    await shot('05_results')

    log('-- remove')
    mat = next(c for c in gui.cases() if c['kind'] == 'mat')
    await page.click(f'#caseBody li[data-id="{mat["id"]}"] input[type="checkbox"]')
    await page.click('#btnRemove')
    await wait_for(lambda: all(c['id'] != mat['id'] for c in gui.cases()), 'removed')
    await check_alive(gui, page, 'remove')

    await pick(gui, page, '#btnAddFiles', OPEN, [os.path.join(d['files'], 'a1_cine.nii.gz')], 'files: again after all that')


async def run(exe, shots):
    from playwright.async_api import async_playwright
    root = tempfile.mkdtemp(prefix='tigerhx_ui_', dir=os.environ.get('RUNNER_TEMP'))
    d = make_data(root)
    gui = Gui(exe, root, d['models'])
    ok = False
    try:
        async with async_playwright() as p:
            browser = None
            for _ in range(60):
                try:
                    browser = await p.chromium.connect_over_cdp(f'http://127.0.0.1:{PORT}')
                    break
                except Exception:
                    await asyncio.sleep(1)
            assert browser, 'cannot connect to the WebView2 debugging port'
            page = None
            for _ in range(60):
                pages = [pg for c in browser.contexts for pg in c.pages if pg.url.startswith(gui.base)]
                if pages:
                    page = pages[0]
                    break
                await asyncio.sleep(0.5)
            assert page, 'the GUI page did not load'
            page.on('pageerror', lambda e: log('PAGE ERROR', e))
            await scenarios(gui, page, d, shots)

            log('-- close the window')
            w = gui.main_window()
            assert w, 'no main window'
            w.close()
            try:
                await browser.close()
            except Exception:
                pass
        gui.proc.wait(timeout=30)
        log('exit code', gui.proc.returncode)
        ok = True
        print('UI test passed', flush=True)
    except BaseException:
        traceback.print_exc()
        if shots:
            try:
                from PIL import ImageGrab
                ImageGrab.grab().save(os.path.join(shots, 'zz_failure_desktop.png'))
            except Exception as e:
                log('no desktop screenshot:', e)
        raise
    finally:
        if gui.proc.poll() is None:
            gui.proc.kill()
        if ok:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    shots = sys.argv[2] if len(sys.argv) > 2 else None
    if shots:
        os.makedirs(shots, exist_ok=True)
    asyncio.run(run(sys.argv[1], shots))
