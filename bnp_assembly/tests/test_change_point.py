import pytest
import numpy as np
from bnp_assembly.change_point import  find_change_point

pairs = [([0, 1, 2, 5, 10, 20, 16, 19], 4),
         ([10, 20, 30, 1, 2, 3][::-1], 3)]


@pytest.mark.parametrize("data,expected", pairs)
def test_find_change_point(data, expected):
    assert find_change_point(data) == expected


def test_find_change_point_real_case():
    data = np.load("missing_data_case.npy")
    data = data[::-1]

    change_point = find_change_point(data)
    correct = 26
    assert abs(change_point-correct) < 2

    no_change_point = data[26:]
    change_point = find_change_point(no_change_point)
    assert change_point == 0

