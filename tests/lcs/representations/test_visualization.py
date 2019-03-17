import pytest

from lcs.representations.visualization import visualize, _scale


@pytest.mark.parametrize("_interval, _range, n, _visualization", [
    ((0, 0), (0, 15), 10, 'O.........'),
    ((0, 1), (0, 15), 10, 'Oo........'),
    ((0, 2), (0, 15), 10, 'Oo........'),
    ((0, 3), (0, 15), 10, 'OOo.......'),
    ((0, 15), (0, 15), 10, 'OOOOOOOOOo'),
    ((4, 13), (0, 15), 10, '..oOOOOOo.'),
    ((5, 13), (0, 15), 10, '...OOOOOo.'),
    ((5, 5), (0, 15), 10, '...O......'),
    ((4, 4), (0, 15), 10, '..o.......'),
])
def test_visualize(_interval, _range, n, _visualization):
    assert visualize(_interval, _range) == _visualization


@pytest.mark.parametrize("_val, _init_n, _n, _result", [
    (2, 4, 10, 5),
    (2, 8, 10, 2),
    (4, 8, 10, 5),
])
def test_scale(_val, _init_n, _n, _result):
    assert _scale(_val, _init_n, _n) == _result
