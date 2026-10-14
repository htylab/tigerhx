"""A workspace holding a list of cases and their analysis state."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from os.path import abspath, isdir, isfile, join, relpath

from tigerhx.core.io import CASE_COLUMNS, read_case_csv, safe_name, strip_nii_ext, write_csv

NII_EXT = ('.nii', '.nii.gz')

# pending -> queued -> running -> review (segmented, apex not confirmed) -> done
STATUSES = ('pending', 'queued', 'running', 'review', 'done', 'error')

DEFAULT_SETTINGS = {
    'model': '',
    'save_mat': True,          # legacy .mat in the workspace
    'mat_in_inputdir': False,  # legacy .mat next to the input file
    'nii_in_inputdir': False,  # label NIfTI next to the input file
    'output_dir': '',          # one folder for every .mat ('' = <workspace>/output)
}


@dataclass
class Case:
    id: str
    path: str
    name: str
    status: str = 'pending'
    progress: float = 0.0
    message: str = ''
    model: str = ''
    shape: list = None
    zooms: list = None
    ed: int = None
    es: int = None
    base_first: bool = None
    apex_slice: int = None
    apex_confirmed: bool = False
    checks: list = None        # slices whose segmentation is worth a look (metrics.slice_checks)
    levels: list = None
    crop: list = None          # [x0, x1, y0, y1] around the heart, for the zoomed view
    outputs: dict = field(default_factory=dict)
    kind: str = 'nifti'        # 'nifti', or 'mat': an original-GUI result opened for viewing
    mat_in_inputdir: bool = None   # per-case output choices from a CSV (None = setting)
    nii_in_inputdir: bool = None

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def folder(self):
        return f'{safe_name(self.name)[:60]}_{self.id[:6]}'


def unique_tmp(path):
    """A temporary name next to path that no other writer uses at the same time."""
    return f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'


def case_id(path):
    return hashlib.sha1(abspath(path).encode('utf-8')).hexdigest()[:12]


def _terms(text):
    return [t.strip().lower() for t in (text or '').replace(';', ',').split(',') if t.strip()]


def scan_results(folder):
    """Original-GUI results in a folder (not its subfolders): *pred*.mat."""
    return sorted(join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith('.mat') and 'pred' in f.lower() and isfile(join(folder, f)))


def scan_folder(folder, include='', exclude=''):
    """NIfTI files below folder whose path relative to folder contains any
    include term (all files if none) and no exclude term (case-insensitive)."""
    inc, exc = _terms(include), _terms(exclude)
    found = []
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(NII_EXT):
                continue
            path = join(root, f)
            rel = relpath(path, folder).lower()
            if inc and not any(t in rel for t in inc):
                continue
            if any(t in rel for t in exc):
                continue
            found.append(path)
    return sorted(found)


class Project:
    def __init__(self, workspace):
        self.workspace = abspath(workspace)
        os.makedirs(join(self.workspace, 'cases'), exist_ok=True)
        self.file = join(self.workspace, 'project.json')
        self.lock = threading.RLock()
        self.cases = OrderedDict()
        self.settings = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if not isfile(self.file):
            return
        with open(self.file, encoding='utf-8') as f:
            d = json.load(f)
        self.settings.update({k: v for k, v in d.get('settings', {}).items()
                              if k in DEFAULT_SETTINGS})
        for cd in d.get('cases', []):
            c = Case.from_dict(cd)
            if c.status in ('queued', 'running'):  # interrupted by a restart
                c.status, c.progress = 'pending', 0.0
            self.cases[c.id] = c

    def save(self):
        with self.lock:
            d = {'settings': self.settings, 'cases': [asdict(c) for c in self.cases.values()]}
            tmp = unique_tmp(self.file)  # another process may save the same workspace
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.file)

    def case_dir(self, case):
        return join(self.workspace, 'cases', case.folder)

    def output_dir(self):
        """The one folder that receives every .mat, as in the original GUI."""
        return abspath(self.settings.get('output_dir') or join(self.workspace, 'output'))

    def csv_dir(self):
        return join(self.workspace, 'csv')

    def _new_case(self, path, name, kind='nifti'):
        cid = case_id(path)
        if cid in self.cases:
            return None
        c = Case(id=cid, path=abspath(path), name=name, kind=kind)
        self.cases[cid] = c
        return c

    def add_paths(self, paths, include='', exclude='', results=False):
        """Add NIfTI files and (recursively) folders, case-list CSVs and
        original-GUI results (.mat; with results=True a folder adds its
        *pred*.mat).  Returns {'added': [cases], 'updated': [cases],
        'missing': [paths named in a CSV that do not exist]}."""
        added, updated, missing = [], [], []
        with self.lock:
            for p in paths:
                p = p.strip().strip('"')
                low = p.lower()
                if isfile(p) and low.endswith('.csv'):
                    a, u, m = self._import_csv(p)
                    added += a
                    updated += u
                    missing += m
                    continue
                if (isfile(p) and low.endswith('.mat')) or (results and isdir(p)):
                    for f in ([p] if isfile(p) else scan_results(p)):
                        c = self._new_case(f, os.path.basename(f)[:-4], kind='mat')
                        if c:
                            added.append(c)
                    continue
                if isdir(p):
                    items = [(f, relpath(f, p)) for f in scan_folder(p, include, exclude)]
                elif isfile(p) and low.endswith(NII_EXT):
                    items = [(p, os.path.basename(p))]
                else:
                    continue
                for path, rel in items:
                    head, tail = os.path.split(rel)
                    name = '/'.join(filter(None, [head.replace(os.sep, '/'), strip_nii_ext(tail)]))
                    c = self._new_case(path, name)
                    if c:
                        added.append(c)
            self.save()
        return {'added': added, 'updated': updated, 'missing': missing}

    def _import_csv(self, csv_path):
        """An original-GUI case list: a filled-in Apex counts as confirmed;
        the two output columns override the settings for that case.  Names
        are relative to the folder the files have in common."""
        rows = read_case_csv(csv_path)
        missing = [r['path'] for r in rows if not isfile(r['path'])]
        rows = [r for r in rows if isfile(r['path'])]
        common = os.path.commonpath([os.path.dirname(r['path']) for r in rows]) if rows else ''
        added, updated = [], []
        for r in rows:
            rel = relpath(r['path'], common)
            head, tail = os.path.split(rel)
            name = '/'.join(filter(None, [head.replace(os.sep, '/'), strip_nii_ext(tail)]))
            c = self._new_case(r['path'], name)
            if c is None:
                c = self.cases[case_id(r['path'])]
                if c.status in ('queued', 'running'):
                    continue
                updated.append(c)
            else:
                added.append(c)
            c.mat_in_inputdir, c.nii_in_inputdir = r['mat_in_inputdir'], r['nii_in_inputdir']
            if r['apex'] is not None:
                c.apex_slice, c.apex_confirmed = r['apex'], True
        return added, updated, missing

    def gen_csv(self, folders, include='', exclude=''):
        """The original GUI's GenCSV: one row per NIfTI found, written to
        <workspace>/csv/files_<time>.csv for filling in the Apex column.
        Returns (path, number of files)."""
        files = []
        for folder in folders:
            folder = folder.strip().strip('"')
            if isdir(folder):
                files += scan_folder(folder, include, exclude)
        os.makedirs(self.csv_dir(), exist_ok=True)
        path = join(self.csv_dir(), f'files_{time.strftime("%y%m%d_%H%M%S")}.csv')
        s = self.settings
        write_csv(path, CASE_COLUMNS, [[f, '', int(bool(s.get('mat_in_inputdir'))),
                                        int(bool(s.get('nii_in_inputdir')))] for f in files])
        return path, len(files)

    def remove(self, ids):
        with self.lock:
            for i in ids:
                c = self.cases.get(i)
                if c is not None and c.status not in ('queued', 'running'):
                    del self.cases[i]
            self.save()
