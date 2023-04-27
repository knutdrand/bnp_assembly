import typing as tp
from scipy.optimize import linear_sum_assignment
from .hic_distance_matrix import Edge, DirectedDistanceMatrix, NodeSide
from .contig_graph import ContigPath
import numpy as np


def best_path(distance_matrix: DirectedDistanceMatrix) -> tp.List[ContigPath]:
    matrix = distance_matrix.data
    matrix = np.pad(matrix, [(0, 2), (0, 2)], constant_values = np.min(matrix)-1)
    np.fill_diagonal(matrix, np.inf)
    row_idx, col_idx = linear_sum_assignment(matrix)
    from_sides = [NodeSide.from_numeric_index(i) for i in row_idx]
    to_sides = [NodeSide.from_numeric_index(i) for i in col_idx]
    edges = [Edge.from_numeric_index(idx)
             for idx in zip(row_idx[:-2], col_idx[:-2])]
    cur_side = to_sides[-2]
    side_dict = dict(zip(from_sides, to_sides))
    ordered_edges = []
    n_sides = len(from_sides)-2
    while True:
        cur_side = cur_side.other_side()
        next_side = side_dict[cur_side]
        if next_side.numeric_index >= n_sides:
            break
        ordered_edges.append(Edge(cur_side, next_side))
        cur_side = next_side
    return [ContigPath.from_edges(ordered_edges)]
