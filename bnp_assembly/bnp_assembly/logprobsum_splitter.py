# Make better distance distribution
# generate matrices
# Hook it into Optimal squares

import numpy as np

from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.contig_map import VectorizedScaffoldMap
from bnp_assembly.distance_distribution import distance_dist, DistanceDistribution
from bnp_assembly.location import LocationPair
from bnp_assembly.max_distance import estimate_max_distance2
from bnp_assembly.square_finder import OptimalSquares, split_based_on_indices, DirectOptimalSquares


class LogsumprobMatrices:
    def __init__(self, size_array, path, distance_distribution):
        self._size_array = size_array
        self._path = path
        self._distance_distribution = distance_distribution
        self._disconnected_matrix = np.zeros((len(size_array), len(size_array)))
        self._connected_matrix = np.zeros((len(size_array), len(size_array)))
        self._genome_size = sum(size_array)
        self._scaffold_map = VectorizedScaffoldMap(path, size_array)

    def register_location_pairs(self, location_pairs):
        mask = self._calculate_distance(location_pairs) < self._distance_distribution.max_distance
        location_pairs = location_pairs.subset_with_mask(mask)
        self._register_for_disconnected(location_pairs)
        self._register_for_connected(location_pairs)

    def _register_for_connected(self, location_pairs: LocationPair):
        i, j = (location_pairs.location_a.contig_id, location_pairs.location_b.contig_id)
        distance_given_connected = self._calculate_distance(location_pairs)
        np.add.at(self._connected_matrix, (i, j), self._logprob_of_distance_given_connected(distance_given_connected))
        np.add.at(self._connected_matrix, (j, i), self._logprob_of_distance_given_connected(distance_given_connected))

    def _register_for_disconnected(self, location_pairs: LocationPair):
        i, j = (location_pairs.location_a.contig_id, location_pairs.location_b.contig_id)
        np.add.at(self._disconnected_matrix, (i, j), self._logprob_of_distance_given_disconnected(location_pairs))
        np.add.at(self._disconnected_matrix, (j, i), self._logprob_of_distance_given_disconnected(location_pairs))

    def _logprob_of_distance_given_connected(self, distance_given_connected):
        return self._distance_distribution.log_probability(distance_given_connected)

    def _logprob_of_distance_given_disconnected(self, distance_given_connected):
        return -np.log(self._distance_distribution.max_distance)

    def _calculate_distance(self, location_pairs):
        return np.abs(
            self._get_global_offset(location_pairs.location_a) - self._get_global_offset(location_pairs.location_b))

    def _get_global_offset(self, location):
        return self._scaffold_map.translate_locations(location)

    @property
    def matrices(self):
        return self._connected_matrix, self._disconnected_matrix


def squares_split(numeric_input_data, path: ContigPath):
    np.seterr(divide='raise')
    size_array = np.array(list(numeric_input_data.contig_dict.values()))
    max_distance = estimate_max_distance2(size_array)
    cumulative_distribution = distance_dist(next(numeric_input_data.location_pairs), numeric_input_data.contig_dict)
    distance_distribution = DistanceDistribution.from_cumulative_distribution(cumulative_distribution, max_distance)
    # distance_distribution = distance_distribution.cut_at_distance(max_distance).normalize()
    matrix_obj = LogsumprobMatrices(size_array, path, distance_distribution)
    for location_pair in next(numeric_input_data.location_pairs):
        matrix_obj.register_location_pairs(location_pair)
    connected_matrix, disconnected_matrix = matrix_obj.matrices
    optimal_squares = DirectOptimalSquares(connected_matrix, disconnected_matrix, max_splits=20)
    splits = optimal_squares.find_splits()
    return split_based_on_indices(path, splits)
