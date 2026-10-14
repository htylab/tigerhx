"""Local HTTP backend for the niivue GUI.

Serves the static page, a small JSON API and display volumes for niivue.
Bound to 127.0.0.1; the API needs the per-launch token, handed to the page
once in the start URL and kept in a SameSite=Strict cookie, so other web
pages cannot drive it.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import secrets
import subprocess
import sys
import threading
import traceback
from collections import OrderedDict
from dataclasses import asdict
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os.path import isdir, isfile, join
from urllib.parse import parse_qs, urlparse

import nibabel as nib
import numpy as np

from tigerhx.core import io, models, pipeline
from tigerhx.core.inference import Cancelled
from tigerhx.core.project import Project

STATIC = join(os.path.dirname(os.path.abspath(__file__)), 'static')
MAX_BODY = 1 << 20


class App:
    """Project state plus a single background worker."""

    def __init__(self, workspace):
        self.project = Project(workspace)
        self.token = secrets.token_urlsafe(24)
        self.jobs = queue.Queue()
        self.cancel = threading.Event()
        self.current = None
        self.model_status = {'state': 'idle', 'name': '', 'progress': 0.0, 'message': ''}
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- state -------------------------------------------------------
    def state(self):
        p = self.project
        with p.lock:
            cases = [asdict(c) for c in p.cases.values()]
            settings = dict(p.settings)
        available = models.list_models()
        choices = sorted(set(available) | set(models.GUI_MODELS))
        if not settings.get('model') or settings['model'] not in choices:
            settings['model'] = available[-1] if available else models.GUI_MODELS[-1]
        return {'cases': cases, 'settings': settings, 'workspace': p.workspace,
                'models': [{'name': m, 'downloaded': m in available} for m in choices],
                'running': self.current, 'queued': self.jobs.qsize(),
                'model_status': dict(self.model_status)}

    def update_settings(self, values):
        p = self.project
        with p.lock:
            for k in ('model', 'gpu', 'save_mat', 'mat_in_inputdir', 'nii_in_inputdir'):
                if k in values:
                    p.settings[k] = values[k] if k == 'model' else bool(values[k])
            p.save()

    # ---- jobs --------------------------------------------------------
    def run(self, ids):
        p = self.project
        with p.lock:
            model, gpu = p.settings.get('model') or models.GUI_MODELS[-1], p.settings.get('gpu')
            targets = ids or [c.id for c in p.cases.values() if c.status in ('pending', 'error')]
            queued = []
            for cid in targets:
                c = p.cases.get(cid)
                if c is None or c.status in ('queued', 'running'):
                    continue
                c.status, c.progress, c.message = 'queued', 0.0, ''
                queued.append(cid)
            p.save()
        self.cancel.clear()
        for cid in queued:
            self.jobs.put((cid, model, gpu))
        return len(queued)

    def stop(self):
        self.cancel.set()
        p = self.project
        while True:
            try:
                cid, _, _ = self.jobs.get_nowait()
            except queue.Empty:
                break
            with p.lock:
                c = p.cases.get(cid)
                if c is not None and c.status == 'queued':
                    c.status = 'pending'
        p.save()

    def _model_progress(self, done, total):
        self.model_status['progress'] = done / total if total else 0.0

    def _worker(self):
        p = self.project
        while True:
            cid, model, gpu = self.jobs.get()
            case = p.cases.get(cid)
            if case is None or case.status != 'queued':
                continue
            self.current = cid
            try:
                if not isfile(join(models.model_dir(), model)):
                    self.model_status.update(state='downloading', name=model, progress=0.0, message='')
                try:
                    model_path = models.ensure_model(model, progress=self._model_progress)
                finally:
                    if self.model_status['state'] == 'downloading':
                        self.model_status['state'] = 'idle'
                with p.lock:
                    case.status, case.progress = 'running', 0.0

                def progress(done, total):
                    case.progress = done / total
                pipeline.segment_case(p, case, model_path, gpu=gpu, progress=progress,
                                      cancel=self.cancel)
                with p.lock:
                    case.status = 'done' if case.apex_confirmed else 'review'
                    case.message = ''
            except Cancelled:
                with p.lock:
                    case.status, case.progress, case.message = 'pending', 0.0, 'stopped'
            except Exception as e:
                traceback.print_exc()
                with p.lock:
                    case.status, case.message = 'error', f'{type(e).__name__}: {e}'
                if self.model_status['state'] == 'downloading' or 'download' in str(e):
                    self.model_status.update(state='error', message=str(e))
            finally:
                self.current = None
                self.invalidate(cid)
                p.save()

    # ---- analysis ----------------------------------------------------
    def set_apex(self, cid, apex_slice, base_first=None):
        p = self.project
        case = p.cases[cid]
        if not pipeline.has_seg(p, case):
            raise ValueError('case is not segmented')
        nz = case.shape[2] if case.shape else None
        if nz is not None and not 0 <= int(apex_slice) < nz:
            raise ValueError(f'apex slice must be within 0..{nz - 1}')
        pipeline.set_apex(p, case, apex_slice, base_first)
        self.invalidate(cid, 'aha')
        p.save()
        # .mat / input-folder files depend on the apex; rewrite them off-thread
        threading.Thread(target=self._write_outputs, args=(case,), daemon=True).start()

    def _write_outputs(self, case):
        try:
            pipeline.write_outputs(self.project, case)
            self.project.save()
        except Exception as e:
            traceback.print_exc()
            with self.project.lock:
                case.message = f'output: {type(e).__name__}: {e}'

    # ---- volumes for the viewer --------------------------------------
    def invalidate(self, cid, kind=None):
        with self._cache_lock:
            for key in [k for k in self._cache if k[0] == cid and (kind is None or k[1] == kind)]:
                del self._cache[key]

    def volume(self, cid, kind):
        key = (cid, kind)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        p = self.project
        case = p.cases[cid]
        if kind == 'image':
            data, _, _, zooms = io.load_nifti(case.path)
            body = io.display_nifti_bytes(data, zooms)
        else:
            fname = {'seg': 'seg.nii.gz', 'aha': 'aha17.nii.gz'}[kind]
            path = join(p.case_dir(case), fname)
            if not isfile(path):
                raise FileNotFoundError(fname)
            labels = np.asarray(nib.load(path).dataobj).astype(np.uint8)
            while labels.ndim < 4:
                labels = labels[..., None]
            body = io.display_nifti_bytes(labels, case.zooms or [1, 1, 1], dtype=np.uint8)
        with self._cache_lock:
            self._cache[key] = body
            while len(self._cache) > 6:
                self._cache.popitem(last=False)
        return body


def open_in_file_manager(path):
    if sys.platform.startswith('win'):
        os.startfile(path)  # noqa: S606
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'tigerhx'

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # ---- helpers -------------------------------------------------
        def _send(self, code, body=b'', ctype='application/json', headers=None):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'))

        def _host_ok(self):
            host = self.headers.get('Host', '')
            port = self.server.server_address[1]
            return host in (f'127.0.0.1:{port}', f'localhost:{port}')

        def _authed(self):
            c = cookies.SimpleCookie(self.headers.get('Cookie', ''))
            return self._host_ok() and 'tigerhx' in c and \
                secrets.compare_digest(c['tigerhx'].value, app.token)

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if n > MAX_BODY:
                raise ValueError('request too large')
            return json.loads(self.rfile.read(n) or b'{}')

        def _static(self, rel):
            path = os.path.normpath(join(STATIC, rel))
            if not path.startswith(STATIC + os.sep) or not isfile(path):
                return self._send(404, b'not found', 'text/plain')
            ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
            if ctype.startswith('text/') or ctype.endswith('javascript'):
                ctype += '; charset=utf-8'
            with open(path, 'rb') as f:
                self._send(200, f.read(), ctype)

        # ---- routes --------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            parts = [x for x in url.path.split('/') if x]
            if not self._host_ok():
                return self._send(403, b'forbidden', 'text/plain')
            if not parts:
                token = parse_qs(url.query).get('token', [''])[0]
                if token and secrets.compare_digest(token, app.token):
                    cookie = f'tigerhx={app.token}; HttpOnly; SameSite=Strict; Path=/'
                    self.send_response(303)
                    self.send_header('Location', '/')
                    self.send_header('Set-Cookie', cookie)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if not self._authed():
                    return self._send(403, 'Open TigerHx from the link printed when it starts.'
                                      .encode(), 'text/plain; charset=utf-8')
                return self._static('index.html')
            if parts == ['favicon.ico']:
                return self._send(204, b'', 'image/x-icon')
            if parts[0] == 'static':
                return self._static('/'.join(parts[1:]))
            if not self._authed():
                return self._json({'error': 'unauthorized'}, 403)
            try:
                if parts == ['api', 'state']:
                    return self._json(app.state())
                if len(parts) == 5 and parts[:2] == ['api', 'cases'] and parts[3] == 'volume':
                    kind = parts[4].split('.')[0]
                    if kind not in ('image', 'seg', 'aha') or parts[2] not in app.project.cases:
                        return self._json({'error': 'not found'}, 404)
                    return self._send(200, app.volume(parts[2], kind), 'application/octet-stream')
            except FileNotFoundError as e:
                return self._json({'error': str(e)}, 404)
            except Exception as e:
                traceback.print_exc()
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
            return self._json({'error': 'not found'}, 404)

        def do_POST(self):
            if not self._authed():
                return self._json({'error': 'unauthorized'}, 403)
            parts = [x for x in urlparse(self.path).path.split('/') if x]
            try:
                body = self._body()
                if parts == ['api', 'cases', 'add']:
                    added = app.project.add_paths(body.get('paths', []), body.get('include', ''),
                                                  body.get('exclude', ''))
                    return self._json({'added': len(added)})
                if parts == ['api', 'cases', 'remove']:
                    app.project.remove(body.get('ids', []))
                    return self._json({'ok': True})
                if parts == ['api', 'run']:
                    return self._json({'queued': app.run(body.get('ids') or [])})
                if parts == ['api', 'stop']:
                    app.stop()
                    return self._json({'ok': True})
                if parts == ['api', 'settings']:
                    app.update_settings(body)
                    return self._json({'ok': True})
                if parts == ['api', 'export']:
                    return self._json({'path': app.project.export_metrics()})
                if parts == ['api', 'open']:
                    cid = body.get('id')
                    path = app.project.workspace
                    if cid in app.project.cases:
                        path = app.project.case_dir(app.project.cases[cid])
                    if isdir(path):
                        open_in_file_manager(path)
                    return self._json({'path': path})
                if len(parts) == 4 and parts[:2] == ['api', 'cases'] and parts[3] == 'apex':
                    if parts[2] not in app.project.cases:
                        return self._json({'error': 'not found'}, 404)
                    app.set_apex(parts[2], body['apex_slice'], body.get('base_first'))
                    return self._json({'ok': True})
            except (ValueError, KeyError) as e:
                return self._json({'error': str(e)}, 400)
            except Exception as e:
                traceback.print_exc()
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
            return self._json({'error': 'not found'}, 404)

        do_HEAD = do_GET

    return Handler


def serve(app, port=0):
    """Start the server on a background thread; returns (server, start_url)."""
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}/?token={app.token}'
