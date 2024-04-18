import numpy as np
import pytest

from bnp_assembly.square_finder import OptimalSquares

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
                      [0, 2, 4, 5]),
                     (np.array([[5, 5, 1, 1, 1],
                                [5, 5, 1, 1, 1],
                                [1, 1, 5, 5, 1],
                                [1, 1, 5, 5, 1],
                                [1, 1, 1, 1, 5]]),
                      [0, 2, 4, 5]),
                     (np.array([[10, 5, 1, 1, 1],
                                [5, 10, 1, 1, 1],
                                [1, 1, 10, 5, 1],
                                [1, 1, 5, 10, 1],
                                [1, 1, 1, 1, 10]]),
                      [0, 2, 4, 5]),
                     (np.array([[10, 5, 1, 1, 1],
                                [5, 10, 1, 1, 1],
                                [1, 1, 10, 5, 3],
                                [1, 1, 5, 10, 5],
                                [1, 1, 3, 5, 10]]),
                      [0, 2, 5])
                     ]


@pytest.mark.skip(reason="Not used")
@pytest.mark.parametrize("matrix_and_truth", matrix_and_truths)
def test_find_splits(matrix_and_truth):
    matrix, truth = matrix_and_truth
    splits = OptimalSquares(matrix, 20).find_splits()
    assert splits == truth
    # assert find_splits(matrix) == truth
