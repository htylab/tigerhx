import glob
import os
import py_compile

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')


@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(ROOT, 'tigerhx', '*.py'))
                                        + glob.glob(os.path.join(ROOT, 'tools', '*.py'))))
def test_modules_compile(path):
    py_compile.compile(path, doraise=True)
