"""The Rust core (rust/tigerhx-core) must give the same results as the
Python core: segmentation, AHA map, slice order and apex, checks, zoom box,
display volumes and the legacy .mat.

Needs the parity example: cargo build --example parity (in rust/), or set
TIGERHX_RUST_PARITY to its path.  Skipped when it is not built.
"""
import json
import os
import subprocess
import sys
from os.path import dirname, isfile, join

import nibabel as nib
import numpy as np
import pytest
from scipy.io import loadmat

sys.path.insert(0, dirname(__file__))
from phantom import phantom, phantom_image, tiny_model  # noqa: E402

from tigerhx.core import io, pipeline  # noqa: E402
from tigerhx.core.project import Project, case_id  # noqa: E402

ROOT = dirname(dirname(os.path.abspath(__file__)))


def parity_bin():
    env = os.environ.get('TIGERHX_RUST_PARITY')
    if env:
        return env
    exe = 'parity.exe' if sys.platform.startswith('win') else 'parity'
    for profile in ('release', 'debug'):
        p = join(ROOT, 'rust', 'target', profile, 'examples', exe)
        if isfile(p):
            return p
    return None


BIN = parity_bin()
pytestmark = pytest.mark.skipif(BIN is None, reason='Rust parity example not built')


def rust(*args):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def tiny_model_3d(path):
    """1x1x1 conv for volume-wise ('xyz') models."""
    import onnx
    from onnx import TensorProto, helper
    w = np.array([0, 1, 2, 3], dtype=np.float32).reshape(4, 1, 1, 1, 1)
    b = np.array([0, -0.25, -0.75, -1.5], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node('Conv', ['x', 'W', 'B'], ['y'])], 'tiny3d',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 'x', 'y', 'z'])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 'x', 'y', 'z'])],
        [helper.make_tensor('W', TensorProto.FLOAT, w.shape, w.ravel()),
         helper.make_tensor('B', TensorProto.FLOAT, b.shape, b)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def save_image(path, img, zooms, dtype=np.float32):
    affine = np.diag(list(zooms[:3]) + [1.0])
    affine[:3, 3] = [-40, 25, 3]
    nimg = nib.Nifti1Image(img.astype(np.float32), affine)
    nimg.header.set_zooms(zooms[:img.ndim])
    nimg.set_data_dtype(dtype)
    nib.save(nimg, path)


def noisy(seg, seed):
    rng = np.random.default_rng(seed)
    img = phantom_image(seg)
    return img + rng.uniform(-90, 90, img.shape).astype(np.float32)


CASES = {
    # name: (image, zooms, file dtype, model kind)
    'base_first': (lambda: phantom_image(phantom()), (1.4, 1.4, 8.0, 1.0), np.int16, '2d'),
    'apex_first': (lambda: phantom_image(phantom(base_first=False)), (1.25, 1.25, 10.0, 1.0), np.float32, '2d'),
    'single_frame': (lambda: phantom_image(phantom(nt=1))[..., 0], (1.0, 1.0, 6.0), np.float32, '2d'),
    'noisy': (lambda: noisy(phantom(nx=80, ny=88, nz=9, nt=5), 1), (1.6, 1.6, 8.0, 1.0), np.int16, '2d'),
    'noisy_xyz': (lambda: noisy(phantom(nz=7, nt=4), 2), (1.3, 1.3, 9.0, 1.0), np.float32, '3d'),
}


@pytest.fixture(scope='module', params=sorted(CASES))
def runs(request, tmp_path_factory):
    name = request.param
    make, zooms, dtype, kind = CASES[name]
    tmp = tmp_path_factory.mktemp(name)
    if kind == '2d':
        model = tiny_model(tmp / 'cine4d_v9001_xy_tiny.onnx')
    else:
        model = tiny_model_3d(tmp / 'cine4d_v9002_xyz_tiny.onnx')
    img = make()
    out = {}
    for impl in ('py', 'rs'):
        d = tmp / impl
        (d / 'input').mkdir(parents=True)
        src = str(d / 'input' / f'{name}.cine.nii.gz')
        save_image(src, img, zooms, dtype)
        ws = str(d / 'ws')
        if impl == 'py':
            p = Project(ws)
            p.settings.update(save_mat=True, mat_in_inputdir=True, nii_in_inputdir=True)
            for c in p.add_paths([src]):
                pipeline.segment_case(p, c, model)
                c.status = 'review'
            p.save()
        else:
            rust('settings', ws, 1, 1, 1)
            rust('run', ws, model, src)
        out[impl] = (ws, src)
    return name, out


def project(ws):
    with open(join(ws, 'project.json'), encoding='utf-8') as f:
        return json.load(f)


def only_case(ws):
    cases = project(ws)['cases']
    assert len(cases) == 1
    return cases[0]


FIELDS = ['name', 'status', 'model', 'shape', 'zooms', 'ed', 'es', 'base_first', 'apex_slice',
          'apex_confirmed', 'checks', 'levels', 'crop', 'message']


def folder(c):
    return c['name'], c['id']


def compare_cases(py_ws, rs_ws):
    pc, rc = only_case(py_ws), only_case(rs_ws)
    for k in FIELDS:
        assert pc[k] == rc[k], k
    assert rc['id'] == case_id(rc['path'])
    assert sorted(pc['outputs']) == sorted(rc['outputs'])
    return pc, rc


def case_dir(ws, c):
    p = Project(ws)
    return p.case_dir(p.cases[c['id']])


def compare_nifti(a, b):
    ia, ib = nib.load(a), nib.load(b)
    assert ia.shape == ib.shape
    assert ia.get_data_dtype() == ib.get_data_dtype()
    np.testing.assert_allclose(ia.affine, ib.affine, atol=1e-5)
    assert ia.header.get_zooms() == ib.header.get_zooms()
    np.testing.assert_array_equal(np.asarray(ia.dataobj), np.asarray(ib.dataobj))


def compare_mat(a, b):
    ma, mb = loadmat(a), loadmat(b)
    keys = sorted(k for k in ma if not k.startswith('__'))
    assert keys == sorted(k for k in mb if not k.startswith('__'))
    for k in keys:
        va, vb = ma[k], mb[k]
        if va.dtype.kind == 'U':
            assert va.tolist() == vb.tolist(), k
            continue
        assert va.dtype == vb.dtype, k
        assert va.shape == vb.shape, k
        np.testing.assert_array_equal(va, vb, err_msg=k)


def compare_all(py_ws, rs_ws):
    pc, rc = compare_cases(py_ws, rs_ws)
    pd, rd = case_dir(py_ws, pc), case_dir(rs_ws, rc)
    assert os.path.basename(rd) == f"{io.safe_name(rc['name'])[:60]}_{rc['id'][:6]}"
    for f in ('seg.nii.gz', 'aha17.nii.gz'):
        compare_nifti(join(pd, f), join(rd, f))
    compare_nifti(pc['outputs']['nii_inputdir'], rc['outputs']['nii_inputdir'])
    for k in ('mat', 'mat_inputdir'):
        assert os.path.basename(pc['outputs'][k]) == os.path.basename(rc['outputs'][k])
        compare_mat(pc['outputs'][k], rc['outputs'][k])
    return pc, rc


def py_display(ws, c, kind, crop):
    """What webgui.server.App.volume serves."""
    p = Project(ws)
    case = p.cases[c['id']]
    if kind == 'image':
        data, _, _, zooms = io.load_nifti(case.path)
    else:
        fname = 'aha17.nii.gz' if kind == 'aha' else 'seg.nii.gz'
        data = np.asarray(nib.load(join(p.case_dir(case), fname)).dataobj).astype(np.uint8)
        while data.ndim < 4:
            data = data[..., None]
        zooms = case.zooms or [1, 1, 1]
    if crop and case.crop:
        x0, x1, y0, y1 = case.crop
        data = data[x0:x1, y0:y1]
    return nib.Nifti1Image.from_bytes(
        io.display_nifti_bytes(data, zooms, dtype=None if kind == 'image' else np.uint8))


def test_segmentation_and_outputs_match(runs):
    _, out = runs
    compare_all(out['py'][0], out['rs'][0])


@pytest.mark.parametrize('kind', ['image', 'seg', 'aha'])
@pytest.mark.parametrize('crop', [False, True])
def test_display_volumes_match(runs, tmp_path, kind, crop):
    _, out = runs
    (py_ws, _), (rs_ws, _) = out['py'], out['rs']
    pc, rc = only_case(py_ws), only_case(rs_ws)
    expected = py_display(py_ws, pc, kind, crop)
    path = tmp_path / 'rs.nii'
    rust('display', rs_ws, rc['id'], kind, int(crop), path)
    got = nib.load(str(path))
    a, b = np.asarray(expected.dataobj), np.asarray(got.dataobj)
    while a.ndim > 3 and a.shape[-1] == 1:
        a = a[..., 0]
    while b.ndim > 3 and b.shape[-1] == 1:
        b = b[..., 0]
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(expected.affine, got.affine, atol=1e-5)


def test_apex_changes_match(runs):
    name, out = runs
    (py_ws, _), (rs_ws, _) = out['py'], out['rs']
    for apex, base_first in ((2, None), (4, False), (3, True)):
        p = Project(py_ws)
        case = next(iter(p.cases.values()))
        pipeline.set_apex(p, case, apex, base_first)
        pipeline.write_outputs(p, case)
        p.save()
        rc = only_case(rs_ws)
        rust('apex', rs_ws, rc['id'], apex, {None: 'none', True: 1, False: 0}[base_first])
        compare_all(py_ws, rs_ws)
