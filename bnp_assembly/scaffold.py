from .hic_distance_matrix import calculate_distance_matrices
from .distance_matrix import DirectedDistanceMatrix
from .forbes_score import calculate_distance_matrices as forbes_matrix
from .forbes_score import get_pair_counts, get_node_side_counts, get_forbes_matrix, get_pscore_matrix
from collections import Counter
from .path_finding import best_path, PathFinder
from .location import LocationPair
from .iterative_join import create_merged_graph
from .networkx_wrapper import PathFinder as nxPathFinder
from .contig_graph import ContigPath
from numpy.testing import assert_array_equal
from .plotting import px
import logging
import numpy as np

PathFinder = nxPathFinder


def split_contig(distance_matrix, path, T=-0.1):
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def scaffold(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', threshold=0.0, **distance_kwargs):
    if distance_measure == 'window':
        original_distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)
        split_matrix=original_distance_matrix
    elif distance_measure == 'forbes':
        pair_counts = get_pair_counts(contig_dict, read_pairs)
        node_side_counts = get_node_side_counts(pair_counts)
        DirectedDistanceMatrix.from_edge_dict(len(contig_dict), pair_counts).plot(level=logging.DEBUG).show()
        original_distance_matrix = get_forbes_matrix(pair_counts, node_side_counts)
        original_distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)

        split_matrix = get_pscore_matrix(pair_counts, node_side_counts)
        split_matrix.plot().show()
        original_distance_matrix.plot().show()

    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    mapping = None
    for _ in range(len(distance_matrix)//2):
        paths = PathFinder(distance_matrix).run()
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            paths = split_contig(split_matrix,
                                 ContigPath.from_node_sides(mapping.popitem()[1]),
                                 T=threshold)


            return paths
    assert len(mapping) == 0, mapping
