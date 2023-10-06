import numpy as np
import pytest

from bnp_assembly.square_finder import find_splits

matrix_and_truths = [(np.array([[5, 5, 0, 0, 0],
                                [5, 5, 0, 0, 0],
                                [0, 0, 5, 5, 5],
                                [0, 0, 5, 5, 5],
                                [0, 0, 5, 5, 5]]),
                                [0, 2, 5]),
                     (np.array([[5, 5, 0, 0, 0],
                                [5, 5, 0, 0, 0],
                                [0, 0, 5, 5, 0],
                                [0, 0, 5, 5, 0],
                                [0, 0, 0, 0, 5]]),
                      [0, 2, 4, 5])]


@pytest.mark.parametrize("matrix_and_truth", matrix_and_truths)
def test_find_splits(matrix_and_truth):
    matrix, truth = matrix_and_truth
    assert find_splits(matrix) == truth
