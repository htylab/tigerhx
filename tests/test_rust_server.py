"""The Rust GUI (rust/tigerhx-gui) serves the same HTTP API as
tigerhx/webgui/server.py; this is test_server.py against its executable.

Needs the built executable: cargo build -p tigerhx-gui (in rust/), or set
TIGERHX_RUST_GUI to its path.  Skipped when it is not built.
"""
import http.cookiejar
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from os.path import dirname, isfile, join

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, dirname(__file__))
from phantom import phantom, phantom_image, tiny_model  # noqa: E402
from test_server import legacy_flow  # noqa: E402

pytest.importorskip('onnx')

ROOT = dirname(dirname(os.path.abspath(__file__)))
MODEL = 'cine4d_v9999_xy_tiny.onnx'


def gui_bin():
    env = os.environ.get('TIGERHX_RUST_GUI')
    if env:
        return env
    exe = 'tigerhxgui.exe' if sys.platform.startswith('win') else 'tigerhxgui'
    for profile in ('release', 'debug'):
        p = join(ROOT, 'rust', 'target', profile, exe)
        if isfile(p):
            return p
    return None


BIN = gui_bin()
pytestmark = pytest.mark.skipif(BIN is None, reason='Rust GUI not built')


def start(workspace, models):
    env = dict(os.environ, TIGERHX_MODEL_DIR=str(models))
    proc = subprocess.Popen([BIN, '--no-open', '-w', str(workspace)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if 'GUI:' in line:
            return proc, line.split('GUI:', 1)[1].strip()
    raise AssertionError(f'no URL printed, exit code {proc.poll()}')


def stop(proc):
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture
def server(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    tiny_model(models / MODEL)
    proc, url = start(tmp_path / 'ws', models)
    _, call, _ = client(url)
    call('/api/settings', {'model': MODEL, 'save_mat': False})
    yield proc, url, models
    stop(proc)


def client(url):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.open(url).read()  # sets the cookie
    base = url.split('/?')[0]

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=60) as r:
            raw = r.read()
            return raw if 'volume' in path else json.loads(raw)
    return opener, call, base


def wait_done(call):
    for _ in range(300):
        st = call('/api/state')
        if all(c['status'] not in ('queued', 'running') for c in st['cases']) and not st['running']:
            return st
        time.sleep(0.1)
    raise AssertionError('segmentation did not finish')


def test_auth_required(server):
    _, url, _ = server
    base = url.split('/?')[0]
    for path in ('/', '/api/state'):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + path)
        assert e.value.code == 403
    req = urllib.request.Request(base + '/api/run', data=b'{}', headers={'Content-Type': 'application/json'})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403
    req = urllib.request.Request(url, headers={'Host': 'evil.example:80'})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403
    # the static files need no token
    with urllib.request.urlopen(base + '/static/app.js') as r:
        assert r.headers['Content-Type'].startswith('text/javascript')
        assert b'tigerhxNative' in r.read()


def test_full_flow(server, tmp_path):
    proc, url, models = server
    opener, call, base = client(url)
    page = opener.open(base + '/').read().decode()
    assert 'niivue.umd.js' in page

    data = tmp_path / 'data'
    data.mkdir()
    seg = phantom()
    nib.save(nib.Nifti1Image(phantom_image(seg).astype(np.int16), np.diag([1.5, 1.5, 8, 1])),
             str(data / 'p1.nii.gz'))
    shutil.copy(data / 'p1.nii.gz', data / 'p1_mask.nii.gz')

    assert call('/api/cases/add', {'paths': [str(data)], 'exclude': 'mask'})['added'] == 1
    assert call('/api/cases/add', {'paths': [str(data)], 'exclude': 'mask'})['added'] == 0
    st = call('/api/state')
    assert [m['name'] for m in st['models'] if m['downloaded']] == [MODEL]
    assert st['settings']['model'] == MODEL and st['settings']['save_mat'] is False
    assert call('/api/run', {})['queued'] == 1
    st = wait_done(call)
    case = st['cases'][0]
    assert case['status'] == 'review', case['message']
    assert case['name'] == 'p1' and case['apex_slice'] == 5 and case['levels'][7] == 'apex'

    for kind, dtype in (('image', np.int16), ('seg', np.uint8), ('aha', np.uint8)):
        img = nib.Nifti1Image.from_bytes(call(f"/api/cases/{case['id']}/volume/{kind}.nii"))
        assert img.shape[:3] == seg.shape[:3] and img.get_data_dtype() == dtype
    vol = nib.Nifti1Image.from_bytes(call(f"/api/cases/{case['id']}/volume/seg.nii"))
    assert (np.asarray(vol.dataobj) == seg).all()

    with pytest.raises(urllib.error.HTTPError) as e:
        call(f"/api/cases/{case['id']}/apex", {'apex_slice': 99})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        call(f"/api/cases/{case['id']}/apex", {})
    assert e.value.code == 400
    call(f"/api/cases/{case['id']}/apex", {'apex_slice': 4})
    case = call('/api/state')['cases'][0]
    assert case['status'] == 'done' and case['apex_confirmed'] and case['levels'][4] == 'apical'
    aha = nib.Nifti1Image.from_bytes(call(f"/api/cases/{case['id']}/volume/aha.nii"))
    assert set(np.unique(np.asarray(aha.dataobj)[:, :, 4])) - {0} <= set(range(13, 17))

    assert case['ed'] == 0 and case['es'] == 3 and case['checks'] == []
    x0, x1, y0, y1 = case['crop']
    for kind in ('image', 'seg', 'aha'):
        img = nib.Nifti1Image.from_bytes(call(f"/api/cases/{case['id']}/volume/{kind}.nii?crop=1"))
        assert img.shape == (x1 - x0, y1 - y0) + seg.shape[2:]
    for path, body in ((f"/api/cases/{case['id']}/phases", {'es': 2}), ('/api/export', {}),
                       ('/api/cases/nope/apex', {'apex_slice': 1}), ('/api/pick', {'kind': 'files'})):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(path, body)
        assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        call('/api/cases/nope/volume/seg.nii')
    assert e.value.code == 404

    # settings: the outputs are written once switched on and the apex changes
    call('/api/settings', {'save_mat': True, 'nii_in_inputdir': 1})
    call(f"/api/cases/{case['id']}/apex", {'apex_slice': 5, 'base_first': True})
    for _ in range(100):
        outputs = call('/api/state')['cases'][0]['outputs']
        if {'mat', 'nii_inputdir'} <= set(outputs):
            break
        time.sleep(0.1)
    assert {'mat', 'nii_inputdir'} <= set(outputs)
    assert isfile(outputs['mat']) and isfile(outputs['nii_inputdir'])
    assert call('/api/open', {'id': 'nope'})['path'].endswith('ws')

    # a case analysed by an older version (no zoom box) is refreshed on start
    stop(proc)
    ws = tmp_path / 'ws'
    with open(ws / 'project.json', encoding='utf-8') as f:
        d = json.load(f)
    d['cases'][0]['crop'] = None
    with open(ws / 'project.json', 'w', encoding='utf-8') as f:
        json.dump(d, f)
    proc2, url2 = start(ws, models)
    try:
        _, call2, _ = client(url2)
        for _ in range(50):
            if call2('/api/state')['cases'][0]['crop'] == case['crop']:
                break
            time.sleep(0.1)
        assert call2('/api/state')['cases'][0]['crop'] == case['crop']
        call2('/api/cases/remove', {'ids': [case['id']]})
        assert call2('/api/state')['cases'] == []
    finally:
        stop(proc2)


def test_stop_requeues(server, tmp_path):
    _, url, _ = server
    _, call, _ = client(url)
    data = tmp_path / 'many'
    data.mkdir()
    img = phantom_image(phantom(nt=12)).astype(np.int16)
    for k in range(4):
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(data / f'c{k}.nii.gz'))
    assert call('/api/cases/add', {'paths': [str(data)]})['added'] == 4
    assert call('/api/run', {})['queued'] == 4
    call('/api/stop', {})
    st = wait_done(call)
    statuses = sorted(c['status'] for c in st['cases'])
    assert set(statuses) <= {'pending', 'review'} and 'pending' in statuses
    assert st['queued'] == 0


def test_legacy_flow(server, tmp_path):
    _, url, _ = server
    _, call, _ = client(url)
    legacy_flow(call, tmp_path)
