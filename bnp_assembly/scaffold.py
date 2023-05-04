from .hic_distance_matrix import calculate_distance_matrices
from collections import Counter
from .path_finding import best_path, PathFinder
from .location import LocationPair
from .iterative_join import create_merged_graph
from .networkx_wrapper import PathFinder as nxPathFinder
from .contig_graph import ContigPath
from numpy.testing import assert_array_equal

import numpy as np

PathFinder = nxPathFinder

def split_contig(distance_matrix, path):
    split_edges = (edge for edge in path.edges if distance_matrix[edge]>=-0.1)
    return path.split_on_edges(split_edges)

def scaffold(contig_dict: dict, read_pairs: LocationPair, window_size=15):
    original_distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, window_size=window_size)
    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    mapping = None
    max_iter = 10
    for _ in range(len(distance_matrix)//2):
        paths = PathFinder(distance_matrix).run()
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            paths = split_contig(original_distance_matrix, ContigPath.from_node_sides(mapping.popitem()[1]))
            return paths
    assert len(mapping) == 0, mapping
