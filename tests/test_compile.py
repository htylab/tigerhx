import glob
import os
import py_compile

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..', 'tigerhx')


@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(ROOT, '*.py'))))
def test_modules_compile(path):
    py_compile.compile(path, doraise=True)
