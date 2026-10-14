"""Validate tigerhx models against ACDC expert labels.

Uses the preprocessed ACDC copy on Hugging Face (mathpluscode/ACDC,
CC BY-NC-SA 4.0: 1 x 1 x 10 mm, 192 x 192 crops around the LV) which has the
4D cine plus labelled ED and ES frames.  Data is downloaded on first use and
never needs to be committed.

    python tools/eval_acdc.py --per-group 2
    python tools/eval_acdc.py --cases patient001,patient061 --models cine4d_v0006_xy_cine.onnx
    python tools/eval_acdc.py --per-group 4 --ed-rule first_frame --out results.csv

Ground-truth volumes/EF are computed from the labels with the same code as the
GUI, so differences come from the segmentation and the ED/ES choice only.
Note: some models (e.g. v0003 '...mms12acdc') were trained on ACDC, which
makes their scores optimistic.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tigerhx.core import metrics, models  # noqa: E402
from tigerhx.core.inference import segment  # noqa: E402

BASE = 'https://huggingface.co/datasets/mathpluscode/ACDC/resolve/main/'
PARTS = ('sax_t', 'sax_ed', 'sax_ed_gt', 'sax_es', 'sax_es_gt')
ACDC_TO_TIGERHX = np.array([0, 3, 2, 1], np.uint8)  # ACDC 1 RV, 3 LV -> tigerhx 1 LV, 3 RV
STRUCTURES = ((1, 'LV'), (2, 'MYO'), (3, 'RV'))


def fetch(path, data_dir):
    dest = os.path.join(data_dir, path)
    if not os.path.isfile(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        models.download(BASE + path, dest)
    return dest


def select_cases(info, cases, per_group):
    if cases:
        return cases.split(',')
    groups = defaultdict(list)
    for pid in sorted(info):
        groups[info[pid]['pathology']].append(pid)
    return [pid for g in sorted(groups) for pid in groups[g][:per_group]]


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float('nan')


def evaluate(pid, info, data_dir, model_paths, ed_rule, use_gt_frames):
    files = {p: fetch(f'train/{pid}/{pid}_{p}.nii.gz', data_dir) for p in PARTS}
    img = nib.load(files['sax_t'])
    data = img.get_fdata(dtype=np.float32)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    nt = data.shape[3]
    gt_ed_f, gt_es_f = int(info['ed_frame']) - 1, int(info['es_frame']) - 1
    gt = {ph: ACDC_TO_TIGERHX[np.asarray(nib.load(files[f'sax_{ph}_gt']).dataobj).astype(int)]
          for ph in ('ed', 'es')}
    ref = metrics.compute_metrics(np.stack([gt['ed'], gt['es']], -1), zooms, 0, 1)

    rows = []
    for mp in model_paths:
        pred = segment(data, mp)
        ed, es = (gt_ed_f, gt_es_f) if use_gt_frames else metrics.find_ed_es(pred, ed_rule)
        m = metrics.compute_metrics(pred, zooms, ed, es)
        cyc = lambda a, b: min(abs(a - b), nt - abs(a - b))  # noqa: E731
        row = {'case': pid, 'group': info['pathology'], 'model': os.path.basename(mp),
               'frames': nt, 'ED_gt': gt_ed_f, 'ED_pred': ed, 'ES_gt': gt_es_f, 'ES_pred': es,
               'ED_off': cyc(ed, gt_ed_f), 'ES_off': cyc(es, gt_es_f)}
        for ph, f in (('ED', gt_ed_f), ('ES', gt_es_f)):
            for label, name in STRUCTURES:
                row[f'Dice_{ph}_{name}'] = round(dice(pred[..., f] == label, gt[ph.lower()] == label), 4)
        for k in ('LV_EF_pct', 'LV_EDV_mL', 'LV_ESV_mL', 'LV_mass_g', 'RV_EF_pct', 'RV_EDV_mL', 'RV_ESV_mL'):
            row[f'{k}_gt'], row[f'{k}_pred'] = ref[k], m[k]
        rows.append(row)
    return rows


def summarize(rows):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)
    for model, R in by_model.items():
        print(f'\n{model}  (n={len(R)})')
        for ph in ('ED', 'ES'):
            d = ' / '.join(f"{name} {np.nanmean([r[f'Dice_{ph}_{name}'] for r in R]):.3f}"
                           for _, name in STRUCTURES)
            print(f'  Dice {ph}: {d}')
        for k in ('LV_EF_pct', 'RV_EF_pct'):
            e = np.array([r[f'{k}_pred'] - r[f'{k}_gt'] for r in R
                          if r[f'{k}_pred'] is not None and r[f'{k}_gt'] is not None])
            print(f'  {k[:5]} error (points): bias {e.mean():+.1f}, SD {e.std():.1f}, '
                  f'MAE {np.abs(e).mean():.1f}, max {np.abs(e).max():.1f}')
        for k in ('LV_EDV_mL', 'LV_mass_g'):
            e = np.array([(r[f'{k}_pred'] - r[f'{k}_gt']) / r[f'{k}_gt'] * 100 for r in R])
            print(f'  {k} error: {e.mean():+.1f}% ± {e.std():.1f}%')
        print(f"  ED within 1 frame: {sum(r['ED_off'] <= 1 for r in R)}/{len(R)}, "
              f"ES within 1 frame: {sum(r['ES_off'] <= 1 for r in R)}/{len(R)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data-dir', default=os.path.join(os.path.expanduser('~'), 'acdc_hf'))
    ap.add_argument('--cases', help='comma-separated ids, e.g. patient001,patient061')
    ap.add_argument('--per-group', type=int, default=2,
                    help='first N cases of each diagnosis group (default 2)')
    ap.add_argument('--models', default='cine4d_v0006_xy_cine.onnx',
                    help='comma-separated model names or paths')
    ap.add_argument('--ed-rule', choices=metrics.ED_RULES, default='max_volume')
    ap.add_argument('--gt-frames', action='store_true',
                    help='use the labelled ED/ES frames (segmentation error only)')
    ap.add_argument('--out', help='per-case CSV')
    args = ap.parse_args(argv)

    with open(fetch('train.csv', args.data_dir), newline='') as f:
        info = {r['pid']: r for r in csv.DictReader(f)}
    model_paths = [models.ensure_model(m) for m in args.models.split(',')]
    rows = []
    for pid in select_cases(info, args.cases, args.per_group):
        print(f'{pid} ({info[pid]["pathology"]}) ...', flush=True)
        rows += evaluate(pid, info[pid], args.data_dir, model_paths, args.ed_rule, args.gt_frames)
    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summarize(rows)


if __name__ == '__main__':
    main()
