import numpy as np
import pytest

from bnp_assembly.orientation_distribution import OrientationDistribution


class DistanceDummy:
    def log_probability(self, distance):
        return np.log(np.asanyarray(distance)*2)


@pytest.fixture
def orientation_distribution():
    return OrientationDistribution(3, 2, DistanceDummy())
    # lambda x: np.log(x * 2))

@pytest.fixture
def double_orientation_distribution():
    return OrientationDistribution([3, 2, 3], [2, 3, 2], DistanceDummy())
    # lambda x: np.log(x * 2))



def test_distance(orientation_distribution):
    assert orientation_distribution.distance(0, 0, 'l', 'l') == 1
    assert orientation_distribution.distance(0, 0, 'l', 'r') == 2
    assert orientation_distribution.distance(0, 0, 'r', 'r') == 4
    assert orientation_distribution.distance(0, 0, 'r', 'l') == 3
    assert orientation_distribution.distance(1, 1, 'r', 'r') == 2
    assert orientation_distribution.distance(1, 1, 'r', 'l') == 3


def test_distance_matrix(orientation_distribution):
    matrix = orientation_distribution.distance_matrix(0, 0)
    np.testing.assert_allclose(matrix, np.array([[1, 2], [3, 4]]))


def test_distribution_matrix(orientation_distribution):
    matrix = orientation_distribution.distribution_matrix(0, 0)
    np.testing.assert_allclose(matrix, np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_distribution_matrix(orientation_distribution):
    matrix = orientation_distribution.distribution_matrix(0, 0)
    np.testing.assert_allclose(matrix, np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_shapes(double_orientation_distribution):
    matrix = double_orientation_distribution.distribution_matrix([0, 0, 0], [0, 0, 0])
    print(matrix)
    assert matrix.shape == (3, 2, 2)

    # np.testing.assert_allclose(matrix, np.array([[[0.1, 0.2], [0.3, 0.4]],
    #                                             [[0.1, 0.3], [0.2, 0.4]]]))


#@pytest.mark.xfail
def test_orientation_distribution(orientation_distribution):
    values = orientation_distribution.orientation_distribution(0, 0).values()
    print(values)
    truth = [1 / 10, 2 / 10, 3 / 10, 4 / 10]
    np.testing.assert_allclose(list(values), truth)

