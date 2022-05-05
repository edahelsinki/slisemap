from slisemap.slisemap import Slisemap
import torch

from .utils import *


def test_save_load(tmp_path):
    sm = get_slisemap(30, 4)
    sm.lbfgs()
    sm.save(tmp_path / "tmp.pt")
    sm2 = Slisemap.load(tmp_path / "tmp.pt")
    assert_allclose(sm.get_L(), sm2.get_L())
    assert_allclose(sm.value(), sm2.value())
    assert_allclose(sm.get_Z(rotate=False), sm2.get_Z(rotate=False))
    assert_allclose(sm.get_B(), sm2.get_B())
    assert_allclose(sm.get_W(), sm2.get_W())
    sm2 = Slisemap.load(tmp_path / "tmp.pt", "cpu")
    assert_allclose(sm.get_L(), sm2.get_L())


def test_anonymous_function_fail(tmp_path):
    sm = get_slisemap(30, 4)
    sm.kernel = lambda x: x
    sm.get_loss_fn()
    try:
        sm.save(tmp_path / "tmp.pt")
    except:
        pass
    assert sm._loss is not None
    assert all_finite(sm.value())


def test_cuda(tmp_path):
    if torch.cuda.is_available():
        sm = get_slisemap(30, 4, cuda=True)
        sm.save(tmp_path / "tmp.pt")
        sm2 = Slisemap.load(tmp_path / "tmp.pt")
        assert_allclose(sm.get_L(), sm2.get_L())
        sm2 = Slisemap.load(tmp_path / "tmp.pt", "cpu")
        assert_allclose(sm.get_L(), sm2.get_L())
