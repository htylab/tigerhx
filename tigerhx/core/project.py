"""A workspace holding a list of cases and their analysis state."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from os.path import abspath, isdir, isfile, join, relpath

from tigerhx.core.io import safe_name, strip_nii_ext

NII_EXT = ('.nii', '.nii.gz')

# pending -> queued -> running -> review (segmented, apex not confirmed) -> done
STATUSES = ('pending', 'queued', 'running', 'review', 'done', 'error')

DEFAULT_SETTINGS = {
    'model': '',
    'save_mat': True,          # legacy .mat in the workspace
    'mat_in_inputdir': False,  # legacy .mat next to the input file
    'nii_in_inputdir': False,  # label NIfTI next to the input file
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

    def add_paths(self, paths, include='', exclude=''):
        """Add files and (recursively) folders; returns the new cases."""
        added = []
        with self.lock:
            for p in paths:
                p = p.strip().strip('"')
                if isdir(p):
                    items = [(f, relpath(f, p)) for f in scan_folder(p, include, exclude)]
                elif isfile(p) and p.lower().endswith(NII_EXT):
                    items = [(p, os.path.basename(p))]
                else:
                    continue
                for path, rel in items:
                    cid = case_id(path)
                    if cid in self.cases:
                        continue
                    head, tail = os.path.split(rel)
                    name = '/'.join(filter(None, [head.replace(os.sep, '/'), strip_nii_ext(tail)]))
                    c = Case(id=cid, path=abspath(path), name=name)
                    self.cases[cid] = c
                    added.append(c)
            self.save()
        return added

    def remove(self, ids):
        with self.lock:
            for i in ids:
                c = self.cases.get(i)
                if c is not None and c.status not in ('queued', 'running'):
                    del self.cases[i]
            self.save()
