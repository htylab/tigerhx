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

    call(f"/api/cases/{case['id']}/phases", {'es': 2})
    call(f"/api/cases/{case['id']}/basal", {'phase': 'ES', 'chamber': 'RV', 'slice': 1})
    case = call('/api/state')['cases'][0]
    assert case['es'] == 2 and case['es_manual'] == 2 and case['basal']['ES']['RV'] == 1
    with pytest.raises(urllib.error.HTTPError) as e:
        call(f"/api/cases/{case['id']}/basal", {'phase': 'ES', 'chamber': 'XX', 'slice': 1})
    assert e.value.code == 400
    call(f"/api/cases/{case['id']}/phases", {'reset': True})
    call('/api/settings', {'ed_rule': 'first_frame'})
    for _ in range(50):  # reanalysis runs in the background
        case = call('/api/state')['cases'][0]
        if case['es'] == 3:
            break
        time.sleep(0.1)
    assert case['ed'] == 0 and case['es'] == 3

    path = call('/api/export', {})['path']
    assert os.path.isfile(path)
