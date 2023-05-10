from .hic_distance_matrix import calculate_distance_matrices
from .forbes_score import calculate_distance_matrices as forbes_matrix
from collections import Counter
from .path_finding import best_path, PathFinder
from .location import LocationPair
from .iterative_join import create_merged_graph
from .networkx_wrapper import PathFinder as nxPathFinder
from .contig_graph import ContigPath
from numpy.testing import assert_array_equal
import plotly.express as px
import numpy as np

PathFinder = nxPathFinder


def split_contig(distance_matrix, path, T=-0.1):
    px.histogram([distance_matrix[e] for e in path.edges], nbins=10).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def scaffold(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', **distance_kwargs):
    if distance_measure == 'window':
        original_distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)
    elif distance_measure == 'forbes':
        original_distance_matrix = forbes_matrix(contig_dict, read_pairs, **distance_kwargs)
    original_distance_matrix.plot().show()
    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    mapping = None
    for _ in range(len(distance_matrix)//2):
        paths = PathFinder(distance_matrix).run()
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            paths = split_contig(original_distance_matrix,
                                 ContigPath.from_node_sides(mapping.popitem()[1]),
                                 T=-0.1 if distance_measure == 'window' else 1
            )

            return paths
    assert len(mapping) == 0, mapping
