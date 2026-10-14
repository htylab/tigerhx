"""Synthetic short-axis cine phantom and a tiny ONNX model for tests."""
import numpy as np


def phantom(nx=96, ny=96, nz=8, nt=6, base_first=True):
    """Label map (x, y, slice, frame): LV cavity shrinking towards the apex,
    5-px myocardium, RV on the +y side (absent in the two apical slices) and
    an apex-cap slice with myocardium only.  ED is frame 0, ES frame nt//2."""
    seg = np.zeros((nx, ny, nz, nt), dtype=np.uint8)
    xx, yy = np.mgrid[:nx, :ny]
    cx, cy = nx / 2, ny / 2 - 8
    r = np.hypot(xx - cx, yy - cy)
    for t in range(nt):
        squeeze = 1 - 0.35 * np.sin(np.pi * t / nt)
        for z in range(nz):
            if z == nz - 1:  # apex cap
                seg[..., z, t][r < 6] = 2
                continue
            r_lv = 14 * (1 - z / nz) * squeeze + 2
            sl = seg[..., z, t]
            sl[r < r_lv] = 1
            sl[(r >= r_lv) & (r < r_lv + 5)] = 2
            if z < nz - 2:
                rv = np.hypot(xx - cx, yy - (cy + r_lv + 12)) < 9 * squeeze + 3
                sl[rv & (sl == 0)] = 3
    return seg if base_first else seg[:, :, ::-1]


INTENSITY = np.array([0.1, 0.35, 0.6, 0.9], dtype=np.float32)


def phantom_image(seg):
    """Image whose per-slice max-normalised intensity the tiny model maps
    back to the labels (needs an RV-bright voxel per slice: the corner)."""
    img = INTENSITY[seg] * 1000
    img[0, 0] = 900  # single bright voxel; removed by the component filter
    return img


def tiny_model(path, name='modelInput'):
    """1x1 conv whose argmax splits [0,1] into quarters -> labels 0..3."""
    import onnx
    from onnx import TensorProto, helper
    w = np.array([0, 1, 2, 3], dtype=np.float32).reshape(4, 1, 1, 1)
    b = np.array([0, -0.25, -0.75, -1.5], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node('Conv', [name, 'W', 'B'], ['modelOutput'])],
        'tiny',
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch_size', 1, 'x', 'y'])],
        [helper.make_tensor_value_info('modelOutput', TensorProto.FLOAT, ['batch_size', 4, 'x', 'y'])],
        [helper.make_tensor('W', TensorProto.FLOAT, w.shape, w.ravel()),
         helper.make_tensor('B', TensorProto.FLOAT, b.shape, b)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)
