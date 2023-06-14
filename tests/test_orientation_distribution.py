import numpy as np
import pytest

from bnp_assembly.orientation_distribution import OrientationDistribution


@pytest.fixture
def orientation_distribution():
    return OrientationDistribution(3, 2, lambda x: np.log(x * 2))


def test_distance(orientation_distribution):
    assert orientation_distribution.distance(0, 0, 'l', 'l') == 1
    assert orientation_distribution.distance(0, 0, 'l', 'r') == 2
    assert orientation_distribution.distance(0, 0, 'r', 'r') == 4
    assert orientation_distribution.distance(0, 0, 'r', 'l') == 3
    assert orientation_distribution.distance(1, 1, 'r', 'r') == 2
    assert orientation_distribution.distance(1, 1, 'r', 'l') == 3


def test_orientation_distribution(orientation_distribution):
    values = orientation_distribution.orientation_distribution(0, 0).values()
    truth = [1 / 10, 2 / 10, 3 / 10, 4 / 10]
    np.testing.assert_allclose(list(values), truth)
