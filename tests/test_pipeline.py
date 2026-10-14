import json
import os
import sys
import threading

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from phantom import phantom, phantom_image, tiny_model  # noqa: E402

pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from scipy.io import loadmat  # noqa: E402

from tigerhx.core import io, pipeline  # noqa: E402
from tigerhx.core.inference import Cancelled, segment  # noqa: E402
from tigerhx.core.project import Project, scan_folder  # noqa: E402


@pytest.fixture
def model(tmp_path):
    return tiny_model(tmp_path / 'cine4d_v9999_xy_tiny.onnx')


@pytest.fixture
def cine(tmp_path):
    seg = phantom()
    d = tmp_path / 'data' / 'sub.01'
    d.mkdir(parents=True)
    path = d / 'sub.01.cine.nii.gz'
    affine = np.diag([1.5, 1.5, 8.0, 1.0])
    nib.save(nib.Nifti1Image(phantom_image(seg).astype(np.int16), affine), str(path))
    nib.save(nib.Nifti1Image(seg, affine), str(d / 'sub.01_mask.nii.gz'))
    return str(path), seg


def test_segment_recovers_phantom(model, cine):
    path, seg = cine
    calls = []
    pred = segment(nib.load(path).get_fdata(), model, progress=lambda d, n: calls.append((d, n)))
    assert pred.dtype == np.uint8 and (pred == seg).all()
    assert calls[-1] == (seg.shape[2] * seg.shape[3],) * 2

    pred3d = segment(nib.load(path).get_fdata()[..., 0], model)
    assert pred3d.shape == seg.shape[:3] and (pred3d == seg[..., 0]).all()


def test_segment_cancel(model, cine):
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        segment(nib.load(cine[0]).get_fdata(), model, cancel=ev)


def test_scan_folder_filters(tmp_path, cine):
    root = str(tmp_path / 'data')
    assert len(scan_folder(root)) == 2
    assert scan_folder(root, exclude='mask') == [cine[0]]
    assert scan_folder(root, include='CINE') == [cine[0]]


def test_project_pipeline(tmp_path, model, cine):
    path, seg = cine
    proj = Project(str(tmp_path / 'ws'))
    proj.settings.update(save_mat=True, nii_in_inputdir=True)
    added = proj.add_paths([str(tmp_path / 'data')], exclude='mask')['added']
    assert [c.name for c in added] == ['sub.01/sub.01.cine']
    case = added[0]

    pipeline.segment_case(proj, case, model)
    assert case.ed == 0 and case.es == 3
    assert case.base_first is True and case.apex_slice == 5 and not case.apex_confirmed
    assert case.checks == [] and case.levels[7] == 'apex'
    assert case.crop == io.crop_box(seg) and case.crop[1] - case.crop[0] < seg.shape[0]

    out = nib.load(os.path.join(proj.case_dir(case), 'aha17.nii.gz'))
    assert np.asarray(out.dataobj).max() == 17
    nii = os.path.join(os.path.dirname(path), 'sub.01.cine_v9999_seg.nii.gz')
    assert (np.asarray(nib.load(nii).dataobj) == seg).all()

    mat = loadmat(case.outputs['mat'])
    for k in ('input_img', 'Seg', 'SegAHA', 'LV', 'LVM', 'RV', 'Seg_crop', 'AHA17',
              'ES_frame_0base', 'ED_frame_0base', 'voxel_size', 'model'):
        assert k in mat
    assert mat['ED_frame_0base'].item() == 0 and mat['ES_frame_0base'].item() == 3

    pipeline.set_apex(proj, case, 4)
    assert case.apex_slice == 4 and case.apex_confirmed and case.status == 'done'

    proj.save()
    again = Project(str(tmp_path / 'ws'))
    assert again.cases[case.id].apex_slice == 4 and again.cases[case.id].crop == case.crop


def test_older_project_file_loads(tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    old = {'settings': {'model': 'm.onnx', 'ed_rule': 'first_frame'},
           'cases': [{'id': 'abc', 'path': '/x/a.nii.gz', 'name': 'a', 'status': 'done',
                      'metrics': {'LV_EF_pct': 60}, 'excluded': {}, 'curves': {}}]}
    (ws / 'project.json').write_text(json.dumps(old), encoding='utf-8')
    proj = Project(str(ws))
    assert proj.cases['abc'].status == 'done' and 'ed_rule' not in proj.settings


def test_cli_run(tmp_path, model, cine):
    import tigerhx
    out = tigerhx.run('', cine[0], str(tmp_path), model=model)
    assert out['output'].endswith('sub.01.cine_hx.nii.gz')
    assert (np.asarray(nib.load(out['output']).dataobj) == cine[1]).all()


def test_guitool_single_row_csv(tmp_path, cine):
    pytest.importorskip('tkinter')
    from tigerhx import guitool

    class Log:
        def insert(self, *a):
            pass

        def see(self, *a):
            pass

    p = str(tmp_path / 'one.csv')
    io.write_csv(p, io.CASE_COLUMNS, [[cine[0], 5, 0, 1]])
    files, options, common = guitool.run_program_gui_interaction(p, Log(), None)
    assert files == [cine[0]] and options[0]['Apex'] == 5 and options[0]['nii_in_inputdir'] == 1
    assert os.path.relpath(files[0], common) == 'sub.01.cine.nii.gz'


def test_concurrent_saves(tmp_path):
    # two windows (or a background thread and the app) saving one workspace
    a, b = Project(str(tmp_path / 'ws')), Project(str(tmp_path / 'ws'))
    errors = []

    def hammer(p):
        try:
            for _ in range(100):
                p.save()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=hammer, args=(p,)) for p in (a, b, a)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert sorted(os.listdir(tmp_path / 'ws')) == ['cases', 'project.json']  # no temp files left


def _cine(path, seg=None, zooms=(1.5, 1.5, 8.0)):
    seg = phantom() if seg is None else seg
    nib.save(nib.Nifti1Image(phantom_image(seg).astype(np.int16), np.diag(list(zooms) + [1])), str(path))
    return seg


def test_csv_import_keeps_apex_and_row_options(tmp_path, model):
    data = tmp_path / 'data'
    (data / 'a').mkdir(parents=True)
    (data / 'b').mkdir()
    _cine(data / 'a' / 'cine.nii.gz')
    _cine(data / 'b' / 'cine.nii.gz')
    csv_path = tmp_path / 'list.csv'
    io.write_csv(str(csv_path), io.CASE_COLUMNS, [
        ['data/a/cine.nii.gz', 4, 0, 1],                 # relative to the CSV, apex confirmed
        [str(data / 'b' / 'cine.nii.gz'), '', '', ''],   # apex left blank: suggested later
        [str(data / 'gone.nii.gz'), 3, 0, 0]])
    proj = Project(str(tmp_path / 'ws'))
    proj.settings.update(save_mat=True, nii_in_inputdir=False)
    r = proj.add_paths([str(csv_path)])
    assert [c.name for c in r['added']] == ['a/cine', 'b/cine']
    assert r['missing'] == [str(data / 'gone.nii.gz')]
    a, b = r['added']
    assert (a.apex_slice, a.apex_confirmed, a.nii_in_inputdir) == (4, True, True)
    assert (b.apex_slice, b.apex_confirmed, b.nii_in_inputdir) == (None, False, None)

    for c in (a, b):
        pipeline.segment_case(proj, c, model)
    assert (a.apex_slice, a.apex_confirmed) == (4, True) and a.levels[4] == 'apical'
    assert (b.apex_slice, b.apex_confirmed) == (5, False)
    # one output folder named as the original GUI did; the row option wins
    assert sorted(os.listdir(proj.output_dir())) == ['a_cine_pred_v9999.mat', 'b_cine_pred_v9999.mat']
    assert os.path.isfile(data / 'a' / 'cine_v9999_seg.nii.gz')
    assert not os.path.exists(data / 'b' / 'cine_v9999_seg.nii.gz')
    sectors = np.asarray(nib.load(os.path.join(proj.case_dir(a), 'sectors.nii.gz')).dataobj)
    assert sectors[..., 4, :].max() == 4 and sectors[..., 3, :].max() == 6

    again = proj.add_paths([str(csv_path)])            # importing again updates the apex
    assert again['added'] == [] and len(again['updated']) == 2


def test_gen_csv(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    for n in ('x_cine.nii.gz', 'x_mask.nii.gz', 'y_cine.nii'):
        (data / n).write_bytes(b'')
    proj = Project(str(tmp_path / 'ws'))
    proj.settings.update(nii_in_inputdir=True)
    path, n = proj.gen_csv([str(data)], exclude='mask')
    rows = io.read_csv(path)
    assert n == 2 and os.path.dirname(path) == proj.csv_dir()
    assert [os.path.basename(r['Filename']) for r in rows] == ['x_cine.nii.gz', 'y_cine.nii']
    assert rows[0]['Apex'] == '' and rows[0]['nii_in_inputdir'] == '1' and rows[0]['mat_in_inputdir'] == '0'


def test_open_original_gui_result(tmp_path, model):
    seg = phantom()
    img = phantom_image(seg).astype(np.float64)
    result = io.legacy_result(img, seg, 5, True, 'cine4d_v9999_xy_tiny.onnx', (1.5, 1.5, 8.0, 1.0))
    out = tmp_path / 'output'
    out.mkdir()
    io.save_mat(str(out / 'sub1_pred_v9999.mat'), result)
    io.save_mat(str(out / 'notes.mat'), {'x': 1})          # not a *pred*.mat: not listed
    proj = Project(str(tmp_path / 'ws'))
    proj.settings.update(save_mat=True, mat_in_inputdir=True, nii_in_inputdir=True)
    added = proj.add_paths([str(out)], results=True)['added']
    assert [(c.name, c.kind) for c in added] == [('sub1_pred_v9999', 'mat')]
    case = added[0]
    pipeline.import_mat_case(proj, case)
    assert (case.apex_slice, case.apex_confirmed, case.base_first) == (5, True, True)
    assert case.model == 'cine4d_v9999_xy_tiny.onnx' and case.zooms[:3] == [1.5, 1.5, 8.0]
    assert (pipeline.load_seg(proj, case) == seg).all()
    data, _, _, zooms = pipeline.load_image(case)
    assert np.allclose(data, img) and zooms[:3] == [1.5, 1.5, 8.0]
    pipeline.write_outputs(proj, case)                     # an opened result writes nothing
    assert case.outputs == {} and sorted(os.listdir(out)) == ['notes.mat', 'sub1_pred_v9999.mat']
    with pytest.raises(ValueError):
        io.load_legacy_mat(str(out / 'notes.mat'))
