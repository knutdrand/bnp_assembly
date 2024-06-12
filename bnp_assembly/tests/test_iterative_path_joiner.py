import logging
import plotly
import plotly.express as px
from bnp_assembly.graph_objects import Edge, NodeSide
from bnp_assembly.sparse_interaction_based_distance import get_edge_counts_with_max_distance

logging.basicConfig(level=logging.INFO)
import pytest
import numpy as np
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.sparse_interaction_matrix import BinnedNumericGlobalOffset, SparseInteractionMatrix
import matplotlib.pyplot as plt


#@pytest.fixture
def matrix():
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes({0: 5, 1: 5, 2: 5, 3: 5}, 1)
    matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            matrix[i, j] = 20 - abs(i-j)

    matrix = SparseInteractionMatrix.from_np_matrix(global_offset, matrix)
    return matrix


def test_iterative_path_joiner_integration():

    n_contigs = 4
    contig_size = 20

    global_offset = BinnedNumericGlobalOffset.from_contig_sizes({i: contig_size for i in range(n_contigs)}, 1)
    matrix = np.zeros((n_contigs * contig_size, n_contigs * contig_size))

    total_size = n_contigs * contig_size

    for i in range(n_contigs * contig_size):
        for j in range(n_contigs * contig_size):
            matrix[i, j] = total_size - abs(i-j)

    # add noise
    #for i in range(n_contigs * contig_size):
    #    for j in range(n_contigs * contig_size):
    #        matrix[i, j] += np.random.random()

    # make symmetric
    for i in range(n_contigs * contig_size):
        for j in range(n_contigs * contig_size):
            matrix[j, i] = matrix[i, j]

    #matrix /= np.max(matrix)
    print(matrix)
    #plotly.express.imshow(matrix, title='matrix').show()
    matrix = SparseInteractionMatrix.from_np_matrix(global_offset, matrix)

    edge_counts, nodeside_sizes = get_edge_counts_with_max_distance(matrix, 2)
    assert edge_counts[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == 79+78+78+77

    edge_counts, nodeside_sizes = get_edge_counts_with_max_distance(matrix, 4)
    correct = 79 + 78 * 2 + 77 * 3 + 76 * 4 + 75 * 3 + 74 * 2 + 73
    assert edge_counts[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == correct
    print("Correct", correct)

    # joiner = IterativePathJoiner(matrix)
    # path = joiner.run(1)
    # px.imshow(joiner._intra_background_means, title="intra background means").show()
    # px.imshow(joiner._intra_background_stds, title="intra background stds").show()
    # px.imshow(joiner._inter_background_means).show()

