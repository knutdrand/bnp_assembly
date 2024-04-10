import logging
logging.basicConfig(level=logging.INFO)
import pytest
import numpy as np
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.sparse_interaction_matrix import BinnedNumericGlobalOffset, SparseInteractionMatrix


#@pytest.fixture
def matrix():
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes({0: 5, 1: 5, 2: 5, 3: 5}, 1)
    matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            matrix[i, j] = 20 - abs(i-j)

    matrix = SparseInteractionMatrix.from_np_matrix(global_offset, matrix)
    return matrix


def test(matrix=matrix()):
    joiner = IterativePathJoiner(matrix)
    path = joiner.run()


if __name__ == "__main__":
    matrix = matrix()
    test(matrix)
