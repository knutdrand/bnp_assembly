import pytest

from bnp_assembly.change_point import  find_change_point

pairs = [([0, 1, 2, 5, 10, 20, 16, 19], 4),
         ([10, 20, 30, 1, 2, 3][::-1], 3)]




@pytest.mark.parametrize("data,expected", pairs)
def test_find_change_point(data, expected):
    assert find_change_point(data) == expected

