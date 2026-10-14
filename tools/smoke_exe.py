"""Smoke test for a built TigerHx GUI executable (CI, after PyInstaller).

Starts the executable without a window, segments a synthetic cine with a
tiny ONNX model through the HTTP API and checks the outputs, so a module
left out of the build shows up as a failure here rather than on a user's
machine.  Needs numpy, nibabel, scipy and onnx in the calling Python.

    python tools/smoke_exe.py dist/tigerhxgui.exe
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import nibabel as nib
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
from phantom import phantom, phantom_image, tiny_model  # noqa: E402

MODEL = 'cine4d_v9999_xy_tiny.onnx'


def main(exe):
    tmp = tempfile.mkdtemp(prefix='tigerhx_smoke_')
    models = os.path.join(tmp, 'models')
    os.makedirs(models)
    tiny_model(os.path.join(models, MODEL))
    cine = os.path.join(tmp, 'p1.nii.gz')
    nib.save(nib.Nifti1Image(phantom_image(phantom()).astype(np.int16), np.diag([1.5, 1.5, 8, 1])), cine)

    env = dict(os.environ, TIGERHX_MODEL_DIR=models)
    proc = subprocess.Popen([exe, '--no-open', '-w', os.path.join(tmp, 'ws')], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = None
        for line in proc.stdout:
            print('|', line.rstrip(), flush=True)
            if 'GUI:' in line:
                url = line.split('GUI:', 1)[1].strip()
                break
        assert url, f'no URL printed, exit code {proc.poll()}'

        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        base = url.split('/?')[0]
        assert 'niivue.umd.js' in opener.open(url).read().decode()

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
            with opener.open(req, timeout=60) as r:
                raw = r.read()
                return raw if '/volume/' in path else json.loads(raw)

        call('/api/settings', {'model': MODEL, 'save_mat': True})
        assert call('/api/cases/add', {'paths': [cine]})['added'] == 1
        call('/api/run', {})
        for _ in range(240):
            case = call('/api/state')['cases'][0]
            if case['status'] not in ('pending', 'queued', 'running'):
                break
            time.sleep(0.5)
        assert case['status'] == 'review', case
        for kind in ('image', 'seg', 'edge', 'aha'):
            assert len(call(f"/api/cases/{case['id']}/volume/{kind}.nii?crop=1")) > 352
        mat = loadmat(case['outputs']['mat'])
        assert all(k in mat for k in ('Seg', 'SegAHA', 'AHA17', 'ED_frame_0base'))
        call(f"/api/cases/{case['id']}/apex", {'apex_slice': 4})
        assert call('/api/state')['cases'][0]['status'] == 'done'
        print('smoke test passed')
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == '__main__':
    main(sys.argv[1])
