from scipy.optimize import linear_sum_assignment
from .hic_distance_matrix import Edge, DirectedDistanceMatrix
from .contig_graph import ContigPath
import numpy as np


def best_path(distance_matrix: DirectedDistanceMatrix):
    matrix = distance_matrix.data
    matrix = np.pad(matrix, [(0, 2), (0, 2)], constant_values = np.min(matrix)-1)
    np.fill_diagonal(matrix, np.inf)
    print(matrix)
    row_idx, col_idx = linear_sum_assignment(matrix)
    edges = [Edge.from_numeric_index(idx)
             for idx in zip(row_idx[:-2], col_idx[:-2])]
    cur_idx = col_idx[len(matrix)-2]
    ordered_edges = []
    print(col_idx)
    while cur_idx < len(matrix)-2:
        ordered_edges.append(edges[cur_idx])
        cur_idx = col_idx[cur_idx]
    print(ordered_edges[:-1])

    return ContigPath.from_edges(ordered_edges[:-1])
