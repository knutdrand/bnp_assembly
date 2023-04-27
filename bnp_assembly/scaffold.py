from .hic_distance_matrix import calculate_distance_matrices
from .path_finding import best_path
from .location import LocationPair
from numpy.testing import assert_array_equal
import numpy as np

def scaffold(contig_dict: dict, read_pairs: LocationPair, window_size=15):
    distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, window_size=window_size)
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    print(np.round(distance_matrix.data.T, 2))
    path = best_path(distance_matrix)
    return path
