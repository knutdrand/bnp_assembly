import random

import numpy as np
import pytest
from shared_memory_wrapper import from_file
import matplotlib.pyplot as plt

from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.make_scaffold import join_all_contigs
from bnp_assembly.sparse_interaction_based_distance import get_distance_matrix_from_sparse_interaction_matrix
from bnp_assembly.sparse_interaction_matrix import BackgroundMatrix
import plotly.express as px


@pytest.mark.skip(reason='slow')
def test_integration_real_case5():
    """
    Case where two contigs need to both be moved to improve the score, i.e. a local optimum where moving one contig
    does not help
    """
    matrix = from_file("interaction_matrix_test6.npz")
    testpath = [DirectedNode(contig, "+") for contig in range(matrix.n_contigs)]
    random.seed(0)
    random.shuffle(testpath)
    matrix = matrix.get_matrix_for_path(testpath, as_raw_matrix=False)
    n_contigs = matrix.n_contigs
    matrix.plot_submatrix(0, matrix.n_contigs - 1)

    background = BackgroundMatrix.from_sparse_interaction_matrix(matrix)
    distance_matrix = get_distance_matrix_from_sparse_interaction_matrix(matrix, background, score_func=None)
    distance_matrix.invert()
    distance_matrix.plot(px=px).show()
    distance_matrix.plot(name="rr_and_ll", dirs='rrll', px=px).show()
    path = join_all_contigs(distance_matrix)
    path = path.directed_nodes
    new_matrix = matrix.get_matrix_for_path(path, as_raw_matrix=False)
    new_matrix.plot_submatrix(0, n_contigs - 1)



