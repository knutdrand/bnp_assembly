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
from .splitting import ScaffoldSplitter, ScaffoldSplitter2, LinearSplitter, ScaffoldSplitter3, LinearSplitter2
import logging
import numpy as np

PathFinder = nxPathFinder


def _split_contig(distance_matrix, path, T=-0.1):
    px('debug').histogram([distance_matrix[edge] for edge in path.edges if distance_matrix[edge]>-0.6], nbins=15).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def split_contig(contig_path, contig_dict, threshold, bin_size, locations_pair):
    return LinearSplitter2(contig_dict,  contig_path).split(locations_pair)
    return ScaffoldSplitter3(contig_dict, bin_size).split(contig_path, locations_pair, threshold)

    # return LinearSplitter(contig_dict, threshold).iterative_split(contig_path, locations_pair)
#     return LinearSplitter(contig_dict).split(contig_path, locations_pair, threshold)


def scaffold(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', threshold=0.0, **distance_kwargs):
    if distance_measure == 'window':
        original_distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)
        split_matrix=original_distance_matrix
    elif distance_measure == 'forbes':
        pair_counts = get_pair_counts(contig_dict, read_pairs)
        node_side_counts = get_node_side_counts(pair_counts)
        DirectedDistanceMatrix.from_edge_dict(len(contig_dict), pair_counts).plot(level=logging.DEBUG).show()
        original_distance_matrix = get_forbes_matrix(pair_counts, node_side_counts)
        split_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)
        # split_matrix = get_pscore_matrix(pair_counts, node_side_counts)
        original_distance_matrix.plot('debug').show()

    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    mapping = None
    for _ in range(len(distance_matrix)//2):
        paths = PathFinder(distance_matrix).run()
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            path = ContigPath.from_node_sides(mapping.popitem()[1])
            paths = split_contig(path, contig_dict, -threshold, 5000, read_pairs)
            return paths
    assert len(mapping) == 0, mapping
