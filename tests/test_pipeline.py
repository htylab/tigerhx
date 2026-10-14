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
    added = proj.add_paths([str(tmp_path / 'data')], exclude='mask')
    assert [c.name for c in added] == ['sub.01/sub.01.cine']
    case = added[0]

    pipeline.segment_case(proj, case, model)
    assert case.ed == 0 and case.es == 3
    assert case.base_first is True and case.apex_slice == 5 and not case.apex_confirmed
    assert case.metrics['LV_EF_pct'] > 30

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
    assert again.cases[case.id].apex_slice == 4

    rows = io.read_csv(proj.export_metrics())
    assert rows[0]['Case'] == 'sub.01/sub.01.cine' and float(rows[0]['LV_EDV_mL']) > 0


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


def test_phase_and_basal_choices(tmp_path, model, cine):
    proj = Project(str(tmp_path / 'ws'))
    proj.settings.update(save_mat=False)
    case = proj.add_paths([cine[0]])[0]
    pipeline.segment_case(proj, case, model)
    ef = case.metrics['LV_EF_pct']

    pipeline.set_phases(proj, case, ed=1)
    assert (case.ed, case.es, case.ed_manual) == (1, 3, 1) and case.metrics['LV_EF_pct'] < ef
    with pytest.raises(ValueError):
        pipeline.set_phases(proj, case, es=99)
    pipeline.set_phases(proj, case, reset=True)
    assert case.ed == 0 and case.ed_manual is None

    rv_esv = case.metrics['RV_ESV_mL']
    pipeline.set_basal(proj, case, 'ES', 'RV', 1)
    assert case.basal['ES']['RV'] == 1 and case.metrics['RV_ESV_mL'] < rv_esv
    pipeline.set_basal(proj, case, 'ES', 'RV', None)
    assert case.metrics['RV_ESV_mL'] == rv_esv
    with pytest.raises(ValueError):
        pipeline.set_basal(proj, case, 'MID', 'RV', 1)

    proj.settings['ed_rule'] = 'first_frame'
    pipeline.set_basal(proj, case, 'ED', 'LV', 2)
    proj.save()
    again = Project(str(tmp_path / 'ws')).cases[case.id]
    assert again.basal == {'ED': {'LV': 2, 'RV': None}, 'ES': {'LV': None, 'RV': None}}
    row = io.read_csv(proj.export_metrics())[0]
    assert row['Basal_ED_LV'] == '2' and row['ED_source'] == 'first_frame'
