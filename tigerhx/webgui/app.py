"""TigerHx GUI launcher.

Runs the local backend and shows the page in a native window when pywebview
is installed (pip install pywebview), otherwise in the default browser.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser

from tigerhx.webgui.server import App, serve


class NativeDialogs:
    """File dialogs exposed to the page as window.pywebview.api.*

    pywebview exposes every public attribute of this object to the page and
    walks into nested objects, so the window is kept private: walking into
    window.native (a WinForms form) recursed without end on Windows.
    """

    _window = None

    def _dialog(self, kind, **kw):
        import webview
        fd = getattr(webview, 'FileDialog', None)
        dialog = getattr(fd, kind) if fd else getattr(webview, f'{kind}_DIALOG')
        return list(self._window.create_file_dialog(dialog, **kw) or [])

    def pick_files(self):
        return self._dialog('OPEN', allow_multiple=True,
                            file_types=('NIfTI, case list or result (*.nii;*.nii.gz;*.csv;*.mat)',
                                        'NIfTI (*.nii;*.nii.gz)', 'Case list (*.csv)',
                                        'TigerHx result (*.mat)', 'All files (*.*)'))

    def pick_folder(self):
        return self._dialog('FOLDER')


def default_workspace():
    return os.environ.get('TIGERHX_WORKSPACE') or os.path.join(os.path.expanduser('~'), 'TigerHx')


def main(argv=None):
    parser = argparse.ArgumentParser(description='TigerHx cine MRI segmentation GUI')
    parser.add_argument('-w', '--workspace', default=default_workspace(),
                        help='folder for results and the case list (default: ~/TigerHx)')
    parser.add_argument('--port', type=int, default=0, help='local port (default: random)')
    parser.add_argument('--browser', action='store_true',
                        help='use the default web browser instead of a native window')
    parser.add_argument('--no-open', action='store_true', help='only print the URL')
    args = parser.parse_args(argv)

    app = App(args.workspace)
    server, url = serve(app, args.port)
    print(f'TigerHx workspace: {app.project.workspace}')
    print(f'TigerHx GUI: {url}', flush=True)

    if not args.browser and not args.no_open:
        try:
            import webview
        except ImportError:
            webview = None
        if webview is not None:
            api = NativeDialogs()
            api._window = webview.create_window('TigerHx', url, js_api=api,
                                                width=1440, height=900, min_size=(1000, 640))
            webview.start()
            app.stop()
            server.shutdown()
            return

    if not args.no_open:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        app.stop()
        server.shutdown()


if __name__ == '__main__':
    sys.exit(main())
