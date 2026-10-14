import os
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from phantom import phantom  # noqa: E402

from tigerhx.core import aha, io, metrics  # noqa: E402
from tigerhx.core.inference import keep_largest_component, model_mode, model_version  # noqa: E402


def legacy_ahaseg(mask, nseg=6):
    """Ray-casting implementation from the original GUI, kept as reference."""
    def mid_to_angles(mid, seg_num):
        if seg_num == 4:
            a = np.array([mid - 135, mid - 45, mid + 45, mid + 135])
        else:
            a = np.array([mid - 120, mid - 60, mid, mid + 60, mid + 120, mid + 180])
        a = (a + 360) % 360
        a = np.append(a, a[0])
        return np.rad2deg(np.unwrap(np.deg2rad(a))).astype(int)

    def circular_sector(theta_range, lvb):
        cx, cy = ndimage.center_of_mass(lvb)
        max_range = np.min(np.abs(lvb.shape - np.array([cx, cy]))).astype(np.int64)
        z = np.arange(0, max_range, 0.1).reshape(-1, 1).dot(
            np.exp(1.0j * theta_range / 180 * np.pi).reshape(1, -1))
        xall = np.round((-np.imag(z) + cx).flatten())
        yall = np.round((np.real(z) + cy).flatten())
        ok = (xall >= 0) & (yall >= 0) & (xall < lvb.shape[0]) & (yall < lvb.shape[1])
        smask = lvb * 0
        smask[xall[ok].astype(int), yall[ok].astype(int)] = 1
        return smask

    lvb, lvw, rvb = mask == 1, mask == 2, mask == 3
    lx, ly = ndimage.center_of_mass(lvb)
    rx, ry = ndimage.center_of_mass(rvb)
    mid = np.angle((rx + 1j * ry) - (lx + 1j * ly)) / np.pi * 180 - 90
    angles = mid_to_angles(mid, nseg)
    sector = lvw * 0
    for ii in range(angles.size - 1):
        sector[circular_sector(np.arange(angles[ii], angles[ii + 1], 0.1), lvb) > 0] = ii + 1
    return sector * lvw


@pytest.mark.parametrize('nseg', [4, 6])
def test_sector_map_matches_original_gui(nseg):
    seg = phantom()
    m = seg[:, :, 2, 0]
    new = aha.sector_map(m, nseg)
    old = legacy_ahaseg(m, nseg)
    both = (new > 0) & (old > 0)
    assert (new > 0).sum() == (m == 2).sum()  # every myocardium pixel labelled
    assert (new[both] == old[both]).mean() > 0.95
    assert set(np.unique(new[m == 2])) == set(range(1, nseg + 1))


def test_sector_map_covers_myocardium_cut_by_image_border():
    m = phantom()[:60, :, 2, 0]  # LV centre at row 48, myocardium clipped at row 59
    assert (aha.sector_map(m, 6) > 0).sum() == (m == 2).sum()
    assert (legacy_ahaseg(m, 6) > 0).sum() < 0.9 * (m == 2).sum()  # original lost pixels


def test_septum_lies_in_sectors_2_and_3():
    m = phantom()[:, :, 2, 0]
    lv = np.argwhere(m == 1).mean(0)
    rv = np.argwhere(m == 3).mean(0)
    myo = np.argwhere(m == 2)
    toward_rv = myo[np.argmin(np.linalg.norm(myo - (lv + 0.4 * (rv - lv)), axis=1))]
    assert aha.sector_map(m, 6)[tuple(toward_rv)] in (2, 3)
    assert aha.sector_map(m, 4)[tuple(toward_rv)] == 2


def test_slice_order_and_apex_suggestion():
    seg = phantom()
    assert aha.detect_base_first(seg, 0) is True
    assert aha.detect_base_first(seg[:, :, ::-1], 0) is False
    # 7 slices with a cavity -> last round(7/3)=2 are apical
    assert aha.suggest_apex_slice(seg, 0, True) == 5
    assert aha.suggest_apex_slice(seg[:, :, ::-1], 0, False) == 2


def test_aha17_levels():
    seg = phantom()
    a = aha.aha17(seg, apex_slice=5, base_first=True, ed=0)
    ed = a[..., 0]
    assert ((ed > 0) == (seg[..., 0] == 2)).all()  # whole myocardium labelled at ED
    labels = [set(np.unique(ed[:, :, z])) - {0} for z in range(seg.shape[2])]
    assert all(lab <= set(range(1, 7)) for lab in labels[0:3])      # basal
    assert all(lab <= set(range(7, 13)) for lab in labels[3:5])     # mid
    assert all(lab <= set(range(13, 17)) for lab in labels[5:7])    # apical (no RV)
    assert labels[7] == {17}                                        # apex cap
    assert labels[0] == set(range(1, 7)) and labels[5] == set(range(13, 17))


def test_aha17_reversed_stack_is_mirrored():
    seg = phantom()
    a = aha.aha17(seg, 5, True, 0)
    b = aha.aha17(seg[:, :, ::-1], 2, False, 0)
    assert (a == b[:, :, ::-1]).all()


def test_legacy_fields():
    seg = phantom().astype(np.int64)
    lvm = aha.legacy_lvm(seg, apex_slice=5)
    assert lvm[:, :, :5].max() == 6 and lvm[:, :, 5:].max() <= 4
    assert lvm[:, :, 7].max() == 1  # missing labels -> whole myocardium = 1
    seg_aha = aha.legacy_seg_aha(seg, lvm)
    assert set(np.unique(seg_aha)) <= {0, 1, 3} | set(range(8, 14))


def test_metrics():
    seg = phantom()
    zooms = [1.5, 1.5, 8.0]
    ed, es = metrics.find_ed_es(seg)
    assert (ed, es) == (0, 3)
    m = metrics.compute_metrics(seg, zooms, ed, es)
    v = 1.5 * 1.5 * 8 / 1000
    lv = (seg == 1).sum(axis=(0, 1, 2)) * v
    assert m['LV_EDV_mL'] == pytest.approx(lv[0], abs=0.01)
    assert m['LV_EF_pct'] == pytest.approx((lv[0] - lv[3]) / lv[0] * 100, abs=0.01)
    assert m['LV_mass_g'] == pytest.approx((seg[..., 0] == 2).sum() * v * 1.05, abs=0.01)
    single = metrics.compute_metrics(seg[..., :1], zooms, 0, 0)
    assert single['LV_EF_pct'] is None and single['LV_EDV_mL'] > 0


def test_keep_largest_component():
    m = np.zeros((40, 40), np.uint8)
    m[5:15, 5:15] = 1   # 100 px
    m[30:32, 30:32] = 1  # 4 px, dropped
    m[20:25, 20:25] = 2  # 25 px < min_size, dropped
    out = keep_largest_component(m, min_size=50)
    assert out.sum() == 100 and out.max() == 1


def test_model_name_parsing():
    assert model_mode('/m/cine4d_v0006_xy_cine.onnx') == 'xy'
    assert model_mode('cine4d_v0002_xyz_mms12acdc.onnx') == 'xyz'
    assert model_version('cine4d_v0006_xy_cine.onnx') == 'v0006'


def test_strip_nii_ext_keeps_dots():
    assert io.strip_nii_ext('/a/sub-01.ses-1.cine.nii.gz') == 'sub-01.ses-1.cine'
    assert io.strip_nii_ext('x.NII') == 'x'


def test_csv_roundtrip_and_encodings(tmp_path):
    p = tmp_path / 'a.csv'
    io.write_csv(p, io.CASE_COLUMNS, [['D:\\資料\\病人,A\\cine.nii.gz', 3, 1, 0]])
    rows = io.read_csv(p)
    assert rows[0]['Filename'] == 'D:\\資料\\病人,A\\cine.nii.gz' and rows[0]['Apex'] == '3'

    p.write_bytes('Filename,Apex\nD:\\資料\\cine.nii.gz,2\n'.encode('cp950'))
    assert io.read_csv(p)[0]['Filename'] == 'D:\\資料\\cine.nii.gz'

    p.write_text('Filename;Apex\n/x/c.nii;4\n', encoding='utf-8')
    assert io.read_csv(p)[0]['Apex'] == '4'


def test_save_labels_and_display_bytes(tmp_path):
    src = nib.Nifti1Image(np.random.rand(8, 6, 3).astype(np.float32), np.diag([2, 2, 5, 1]))
    src.header.set_slope_inter(2.0, 1.0)
    labels = np.random.randint(0, 4, (8, 6, 3))
    io.save_labels(labels, src.affine, src.header, str(tmp_path / 's.nii.gz'))
    back = nib.load(str(tmp_path / 's.nii.gz'))
    assert back.get_data_dtype() == np.uint8
    assert (np.asarray(back.dataobj) == labels).all()

    data = np.arange(8 * 6 * 3 * 2, dtype=np.float32).reshape(8, 6, 3, 2)
    img = nib.Nifti1Image.from_bytes(io.display_nifti_bytes(data, [1.0, 2.0, 5.0]))
    assert img.get_data_dtype() == np.int16 and img.shape == data.shape
    # voxel row i points to -Y (down the screen), column j to +X (right)
    assert (img.affine[:3, :3] @ [1, 0, 0] == [0, -1, 0]).all()
    assert (img.affine[:3, :3] @ [0, 1, 0] == [2, 0, 0]).all()


def test_ed_rules_and_basal_exclusion():
    seg = phantom()
    assert metrics.find_ed_es(seg, 'first_frame') == (0, 3)
    rolled = np.roll(seg, 2, axis=3)                # largest LV volume now at frame 2
    assert metrics.find_ed_es(rolled) == (2, 5)
    assert metrics.find_ed_es(rolled, 'first_frame') == (0, 5)
    with pytest.raises(ValueError):
        metrics.find_ed_es(seg, 'r_wave')

    zooms = [1.5, 1.5, 8.0]
    v = 1.5 * 1.5 * 8 / 1000
    full = metrics.compute_metrics(seg, zooms, 0, 3)
    rv = (seg == 3).sum(axis=(0, 1, 2)) * v
    assert full['RV_EDV_mL'] == pytest.approx(rv[0], abs=0.01)   # RV at the ED/ES frames
    assert full['RV_ESV_mL'] == pytest.approx(rv[3], abs=0.01)

    excluded = {'ED': {'LV': [], 'RV': []}, 'ES': {'LV': [], 'RV': [0]}}
    cut = metrics.compute_metrics(seg, zooms, 0, 3, excluded)
    assert cut['RV_ESV_mL'] == pytest.approx(rv[3] - (seg[:, :, 0, 3] == 3).sum() * v, abs=0.01)
    assert cut['RV_EDV_mL'] == full['RV_EDV_mL'] and cut['LV_EF_pct'] == full['LV_EF_pct']
    assert list(metrics.included_slices(4, [0, 3])) == [False, True, True, False]


def test_slice_checks():
    seg = phantom()
    kinds = {(c['kind'], c['chamber'], c['slice']) for c in metrics.slice_checks(seg, 0, 3)}
    assert kinds == {('basal', 'LV', 0), ('basal', 'RV', 0)}     # clean phantom: reminders only
    bad = seg.copy()
    bad[30:60, 30:60, 7, 3] = 1          # big 'LV' below the apex at ES only
    bad[..., 3, 0][bad[..., 3, 0] == 1] = 0  # LV missing at ED on a middle slice
    kinds = {(c['kind'], c['phase'], c['chamber'], c['slice']) for c in metrics.slice_checks(bad, 0, 3)}
    assert ('es_gt_ed', 'ES', 'LV', 7) in kinds and ('gap', 'ED', 'LV', 3) in kinds
    skip = {'ED': {'LV': [4]}}  # a neighbour the user excluded is no evidence of a gap
    kinds = {(c['kind'], c['phase'], c['chamber'], c['slice']) for c in metrics.slice_checks(bad, 0, 3, True, skip)}
    assert ('gap', 'ED', 'LV', 3) not in kinds
    assert metrics.slice_checks(seg[..., :1], 0, 0) == []
