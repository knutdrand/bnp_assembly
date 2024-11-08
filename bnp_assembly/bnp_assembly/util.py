import numpy as np

from bnp_assembly.graph_objects import Edge


def add_dict_counts(dict_a, dict_b):
    return {key: dict_a.get(key, 0) + dict_b.get(key, 0) for key in set(dict_a) | set(dict_b)}


def get_all_possible_edges(n_contigs):
    all_edges = (Edge.from_numeric_index((i, j)) for i in range(n_contigs*2) for j in range(n_contigs*2))
    return (edge for edge in all_edges if edge.from_node_side.node_id != edge.to_node_side.node_id)




def logaddexp_high_precision(a, b):
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))


def logaddexp_high_precision_arrays(a, b):
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length")
    return np.array([logaddexp_high_precision(a_val, b_val) for a_val, b_val in zip(a, b)])