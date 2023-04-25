from scipy.optimize import linear_sum_assignment
import numpy as np


def best_path(graph):
    n = len(graph._nodes)
    distance_matrix = np.zeros((2+2*n, 2+2*n))
    matrices = [[graph._ll_edges, graph._rl_edges.T],
                [graph._rl_edges, graph._rr_edges]]
    for from_dir in (0, 1):
        for to_dir in (0, 1):
            distance_matrix[from_dir*n:from_dir*n+n, to_dir*n:to_dir*n+n] = matrices[from_dir][to_dir]
    np.fill_diagonal(distance_matrix, np.inf)
    row_idx, col_idx = linear_sum_assignment(distance_matrix)
    cur_idx = col_idx[-1]
    path = []
    seen = set()
    node_ids = []
    reverse_mask = []
    while cur_idx < 2*n:
        print(cur_idx)
        node_id, strand = (cur_idx % n, cur_idx//n)
        if node_id in seen:
            break
        node_ids.append(node_id)
        reverse_mask.append(strand)
        seen.add(node_id)
        path.append((graph._nodes[node_id], strand))
        cur_idx = col_idx[(1-strand)*n+node_id]
    return ContigPath(node_ids, reverse_mask, graph.nodes)
