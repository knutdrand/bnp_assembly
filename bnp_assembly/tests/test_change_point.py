import pytest
import numpy as np
from bnp_assembly.change_point import  find_change_point
import matplotlib.pyplot as plt

from bnp_assembly.missing_data import find_clips

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


def test_find_change_point_real_case2():
    data = np.load("missing_data_case_bin_size1000.npy")
    change_point = find_change_point(data)
    print("CHange point", change_point)
    #plt.plot(data)
    #plt.show()

    clips = find_clips(data, None, None)
    print(clips)