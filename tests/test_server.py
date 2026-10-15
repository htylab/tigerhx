import http.cookiejar
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from phantom import phantom, phantom_image, tiny_model  # noqa: E402

pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

MODEL = 'cine4d_v9999_xy_tiny.onnx'


@pytest.fixture
def server(tmp_path, monkeypatch):
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    tiny_model(models_dir / MODEL)
    monkeypatch.setenv('TIGERHX_MODEL_DIR', str(models_dir))
    from tigerhx.webgui.server import App, serve
    app = App(str(tmp_path / 'ws'))
    app.update_settings({'model': MODEL, 'save_mat': False})
    srv, url = serve(app)
    yield app, url
    app.stop()
    srv.shutdown()


def client(url):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    base = url.split('/?')[0]

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req) as r:
            raw = r.read()
            return raw if 'volume' in path else json.loads(raw)
    return opener, call, base


def test_native_api_exposes_only_the_dialogs():
    # pywebview hands every public attribute of js_api to the page and walks
    # into nested objects; a public window led into WinForms objects whose
    # properties never end (Bounds.Empty.Empty...), so only methods may be public.
    import inspect
    from tigerhx.webgui.app import NativeDialogs
    api = NativeDialogs()
    api._window = object()
    public = {n: getattr(api, n) for n in dir(api) if not n.startswith('_')}
    assert set(public) == {'pick_files', 'pick_folder'}
    assert all(inspect.ismethod(f) for f in public.values())


def test_native_dialogs(monkeypatch):
    # pywebview's own check of file types (webview.util.parse_file_type): a
    # description with a comma made the whole file dialog fail
    import re
    from tigerhx.webgui.app import FILE_TYPES, NativeDialogs
    pattern = r'^([\w ]+)\((\*(?:\.(?:\w+|\*))*(?:;\*(?:\.(?:\w+|\*))*)*)\)$'
    assert all(re.search(pattern, f) for f in FILE_TYPES)

    import types
    monkeypatch.setitem(sys.modules, 'webview', types.SimpleNamespace(
        FileDialog=types.SimpleNamespace(OPEN=10, FOLDER=20)))

    class Window:  # no WinForms form: the dialog opens on the calling thread
        def create_file_dialog(self, dialog, **kw):
            self.got = (dialog, kw)
            return ('a.nii.gz',)
    api = NativeDialogs()
    api._window = Window()
    assert api.pick_files() == ['a.nii.gz'] and api._window.got[1]['file_types'] == FILE_TYPES
    assert api.pick_folder() == ['a.nii.gz']


def test_auth_required(server):
    app, url = server
    base = url.split('/?')[0]
    for path in ('/', '/api/state'):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + path)
        assert e.value.code == 403
    req = urllib.request.Request(url, headers={'Host': 'evil.example:80'})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403


def test_full_flow(server, tmp_path):
    app, url = server
    opener, call, base = client(url)
    page = opener.open(url).read().decode()  # sets the cookie, then serves the page
    assert 'niivue.umd.js' in page

    data = tmp_path / 'data'
    data.mkdir()
    seg = phantom()
    nib.save(nib.Nifti1Image(phantom_image(seg).astype(np.int16), np.diag([1.5, 1.5, 8, 1])),
             str(data / 'p1.nii.gz'))
    shutil.copy(data / 'p1.nii.gz', data / 'p1_mask.nii.gz')

    assert call('/api/cases/add', {'paths': [str(data)], 'exclude': 'mask'})['added'] == 1
    assert call('/api/run', {})['queued'] == 1
    for _ in range(100):
        st = call('/api/state')
        if st['cases'][0]['status'] not in ('queued', 'running'):
            break
        time.sleep(0.1)
    case = st['cases'][0]
    assert case['status'] == 'review', case['message']
    assert case['apex_slice'] == 5 and case['levels'][7] == 'apex'

    for kind, dtype in (('image', np.int16), ('seg', np.uint8), ('aha', np.uint8)):
        img = nib.Nifti1Image.from_bytes(call(f"/api/cases/{case['id']}/volume/{kind}.nii"))
        assert img.shape == seg.shape and img.get_data_dtype() == dtype
    vol = nib.Nifti1Image.from_bytes(call(f"/api/cases/{case['id']}/volume/seg.nii"))
    assert (np.asarray(vol.dataobj) == seg).all()

    with pytest.raises(urllib.error.HTTPError) as e:
        call(f"/api/cases/{case['id']}/apex", {'apex_slice': 99})
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
    for path, body in ((f"/api/cases/{case['id']}/phases", {'es': 2}), ('/api/export', {})):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(path, body)
        assert e.value.code == 404

    # a case analysed by an older version (no zoom box) is refreshed on start
    with app.project.lock:
        app.project.cases[case['id']].crop = None
        app.project.save()
    from tigerhx.webgui.server import App
    again = App(app.project.workspace)
    for _ in range(50):
        if again.project.cases[case['id']].crop == case['crop']:
            break
        time.sleep(0.1)
    assert again.project.cases[case['id']].crop == case['crop']


def wait_idle(call, n=100):
    for _ in range(n):
        st = call('/api/state')
        if all(c['status'] not in ('queued', 'running') for c in st['cases']):
            return st
        time.sleep(0.1)
    raise AssertionError('still running')


def legacy_flow(call, tmp_path):
    """The original GUI's ways in: GenCSV, a CSV with the apex filled in, its
    results opened again, one output folder, the log.  Shared with the Rust
    server test."""
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('p1', 'p2'):
        nib.save(nib.Nifti1Image(phantom_image(phantom()).astype(np.int16), np.diag([1.5, 1.5, 8, 1])),
                 str(data / f'{name}.nii.gz'))
    out = tmp_path / 'out'
    call('/api/settings', {'output_dir': str(out), 'save_mat': True})
    gen = call('/api/gencsv', {'paths': [str(data)], 'include': '', 'exclude': ''})
    assert gen['count'] == 2 and os.path.isfile(gen['path'])
    with open(gen['path'], encoding='utf-8-sig') as f:
        rows = f.read().splitlines()
    assert rows[0] == 'Filename,Apex,mat_in_inputdir,nii_in_inputdir' and len(rows) == 3
    # fill in one apex, as a user would in Excel
    rows[1] = rows[1].replace(',,', ',4,', 1)
    rows[1] = rows[1][:rows[1].rfind(',')] + ',1'
    with open(gen['path'], 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(rows) + '\n')

    r = call('/api/cases/add', {'paths': [gen['path']]})
    assert (r['added'], r['updated'], r['missing']) == (2, 0, [])
    st = call('/api/state')
    assert st['output_dir'] == str(out)
    by_name = {c['name']: c for c in st['cases']}
    assert by_name['p1']['apex_slice'] == 4 and by_name['p1']['apex_confirmed']
    assert by_name['p1']['nii_in_inputdir'] is True and by_name['p2']['nii_in_inputdir'] is False
    assert call('/api/run', {})['queued'] == 2
    st = wait_idle(call)
    by_name = {c['name']: c for c in st['cases']}
    assert by_name['p1']['status'] == 'done' and by_name['p1']['apex_slice'] == 4
    assert by_name['p2']['status'] == 'review'
    assert sorted(os.listdir(out)) == ['p1_pred_v9999.mat', 'p2_pred_v9999.mat']
    assert os.path.isfile(data / 'p1_v9999_seg.nii.gz') and not os.path.exists(data / 'p2_v9999_seg.nii.gz')
    sec = nib.Nifti1Image.from_bytes(call(f"/api/cases/{by_name['p1']['id']}/volume/sectors.nii"))
    sec = np.asarray(sec.dataobj)
    assert sec[:, :, 4].max() == 4 and sec[:, :, 3].max() == 6

    # the results folder opened again: read-only cases with the CSV's apex
    r = call('/api/cases/add', {'paths': [str(out)], 'results': True})
    assert r['added'] == 2
    st = wait_idle(call)
    opened = {c['name']: c for c in st['cases'] if c['kind'] == 'mat'}
    assert set(opened) == {'p1_pred_v9999', 'p2_pred_v9999'}
    p1 = opened['p1_pred_v9999']
    assert p1['status'] == 'done' and p1['apex_slice'] == 4 and p1['base_first'] is True
    img = nib.Nifti1Image.from_bytes(call(f"/api/cases/{p1['id']}/volume/image.nii"))
    assert img.shape == phantom().shape
    assert call('/api/run', {'ids': [p1['id']]})['queued'] == 0   # nothing to re-run
    assert sorted(os.listdir(out)) == ['p1_pred_v9999.mat', 'p2_pred_v9999.mat']

    log = call('/api/log?after=0')['entries']
    texts = ' | '.join(e['text'] for e in log)
    assert 'GenCSV' in texts and 'Finished p1' in texts and 'Opened p1_pred_v9999' in texts
    last = log[-1]['id']
    assert call(f'/api/log?after={last}')['entries'] == []


def test_legacy_flow(server, tmp_path):
    app, url = server
    opener, call, base = client(url)
    opener.open(url).read()
    legacy_flow(call, tmp_path)


def test_download_all_models(server, monkeypatch):
    app, url = server
    opener, call, base = client(url)
    opener.open(url).read()
    from tigerhx.core import models
    got = []

    def fake(name, progress=None):
        got.append(name)
        open(os.path.join(models.model_dir(), name), 'wb').close()
    monkeypatch.setattr(models, 'ensure_model', fake)
    assert call('/api/models/download', {})['downloading'] == len(models.GUI_MODELS)
    for _ in range(50):
        if len(got) == len(models.GUI_MODELS) and app.model_status['state'] == 'idle':
            break
        time.sleep(0.1)
    assert got == models.GUI_MODELS
    assert call('/api/models/download', {})['downloading'] == 0
    assert all(m['downloaded'] for m in call('/api/state')['models'])
