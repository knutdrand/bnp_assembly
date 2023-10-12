# Make better distance distribution
# generate matrices
# Hook it into Optimal squares
import dataclasses
import logging
from typing import List

import more_itertools
import numpy as np
import plotly.express as px
from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.contig_map import VectorizedScaffoldMap
from bnp_assembly.distance_distribution import distance_dist, DistanceDistribution
from bnp_assembly.location import LocationPair
from bnp_assembly.max_distance import estimate_max_distance2
from bnp_assembly.square_finder import OptimalSquares, split_based_on_indices, DirectOptimalSquares

logger  = logging.getLogger(__name__)

'''
Try to find a splitting of the contig path which makes the joined likelihood of the observed reads as large as possible
Only look at reads that lies within {max_distance} of the diagonal.
Assume reads outside of the range are uniformly distributed
Assume reads inside of a scaffold and with the range are distributed according to the distance distribution
Assume reads between scaffolds are distributed uniformly.
Probabilities for reads are then

P(pair) = P(pair in range)*P(pair in scaffold| pair in range) P(d(pair) | pair in range and pair in scaffold) if inside
P(pair) = P(pair in range)*P(pair not in scaffold | pair in range)*Uniform(B) if oustside scaffold
P(Pair) = P(pair not in range) if outside range

R := in range
S := in scaffold
L := location pair

R ~ Bernoulli(r) where rhat = n_reads inside/n_reads
S | R ~ Bernoulli(s) where shat =  n_reads inside scaffold / n_reads inside range
L | S ~ DistanceDist(d(L))/sum(DistanceDist(d(l)) for l in scaffold) ## Maybe approximate this with something
L | not S ~ 1/area(not S)
L | not R ~ 1/area(not R)

Which of these probabilities are dependent on the split?
S | R
L | S in the normalization constant
L | not S in the area

P(L | R) = P(S|R)*P(L|S) if S else P(not S| R) * P(L|S)

P(all(L==l for l in locations) = P(all((L==l for l in locations if cell_ij[l]) for ij in contig_pairs)

C_ij = prod(DistanceDist(d(l))/N for l in locations_ij) = prod(DistanceDist(d(l)) for l in location_ij)/N**n_reads
N = sum(DistanceDist(d(i,j) for i, j in scaffolds) = sum(N_cell for cell in scaffold)
D_ij = prod(1/A for location in locations) = 1/A**n_reads
A = sum(1 for i, j in not scaffolds)

Strategy for reads:
Count reads:
Outside range
Inside range
Inside range per cell
DistanceDist weighted counts per cell

Pre calculation:
Calculate each cells contribution to N (n_ij)
Calculate eacch cells contribution to A (a_ij).

For each split, calculate N as the sum(n_ij for ij in scaffold)
calculate A as the sum(a_ij for ij not in scaffold)
'''

class SplitScorer:
    def __init__(self, N_matrix, A_matrix, ):


class BaseProbabilityMatrices:
    def __init__(self, size_array, path, distance_distribution):
        self._size_array = size_array
        self._path = path
        self._distance_distribution = distance_distribution
        self._disconnected_matrix = np.zeros((len(size_array), len(size_array)))
        self._connected_matrix = np.zeros((len(size_array), len(size_array)))
        self._genome_size = sum(size_array)
        self._scaffold_map = VectorizedScaffoldMap(path, size_array)
        self._inside_matrix = np.zeros((len(size_array), len(size_array)))

    def fill_inside_matrix(self):
        offsets = np.insert(np.cumsum(self._size_array), 0, 0)
        for i, size_a in enumerate(self._size_array):
            for j, size_b in enumerate(self._size_array):
                shape = (size_a, size_b)
                offset = abs(offsets[i] - offsets[j])
                self._inside_matrix[i, j] = self.calculate_inside(offset, shape)

    def fill_outside_matrix(self):
        pass

    def calculate_outside(self, offset, shape):
        s = 0
        for i in range(shape[0]+shape[1]-1)
            if offset+i> self._max_distance:
                break
            n_cells = min(i+1, shape[0], shape[1], sum(shape)-i-1)
            prob = 1/1
            s+=prob*n_cells # Should add real probs here
        return s

    def calculate_inside(self, offset, shape):
        s = 0
        for i in range(shape[0]+shape[1]-1)
            if offset+i> self._max_distance:
                break
            n_cells = min(i+1, shape[0], shape[1], sum(shape)-i-1)
            prob = self._distance_distribution.log_probability(offset)
            s+=prob*n_cells # Should add real probs here
        return s


class LogsumprobMatrices:
    def __init__(self, size_array, path, distance_distribution):
        self._size_array = size_array
        self._path = path
        self._distance_distribution = distance_distribution
        self._genome_size = sum(size_array)
        self._scaffold_map = VectorizedScaffoldMap(path, size_array)

        self._disconnected_matrix = np.zeros((len(size_array), len(size_array)))
        self._connected_matrix = np.zeros((len(size_array), len(size_array)))


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

@dataclasses.dataclass
class EstimationData:
    outside_range_counts: int
    inside_range_counts: int
    inside_range_per_cell: np.ndarray
    distance_wighted_counts: np.ndarray
    inside_normalization: np.ndarray
    outside_normalization: np.ndarray

    def score_split(self, split_indices: List[int]):
        intervals = more_itertools.pairwise(split_indices)
        inside_indices = [slice(*interval) for interval in intervals]
        outside_indices = [(slice(start, end), slice(None, start)) for start, end in intervals] # needs also the transpose

        p_S_given_R = sum(self.inside_range_per_cell[i] for i in inside_indices)/self.inside_range_per_cell
        p_not_S_given_R = 1-p_S_given_R

        N = sum(self.inside_normalization[i] for i in inside_indices)
        A = sum(self.outside_normalization[i]+self.outside_normalization.T[i] for i in inside_indices)




        scores = [self.get_inside_triangle(start, end, self._connected_logprobs) for start, end in intervals]
        scores2 = [self.get_outside_cells(start, end, self._disconnected_log_probs) for start, end in intervals]
        score = sum(s.sum() for s in scores + scores2)
        print(f'Score {score} {split_indices}')
        return score

class CountEverything(LogsumprobMatrices):
    '''
    Strategy:
    Count reads:
    Outside range
    Inside range
    Inside range per cell
    DistanceDist weighted counts per cell
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._outside_range_counter = 0
        self._inside_range_counter = 0
        self._n_nodes = len(self._size_array)
        self._inside_range_per_cell = np.zeros((self._n_nodes, self._n_nodes), dtype=int)
        self._distance_dist_weighted_counts = np.zeros((self._n_nodes, self._n_nodes))

    def register_location_pairs(self, location_pairs):
        dists = self._calculate_distance(location_pairs)
        mask= dists<self._max_dist
        n_inside = np.count_nonzero(mask)
        self._inside_range_counter+=n_inside
        self._outside_range_counter+= len(mask)-n_inside
        location_pairs = location_pairs.subset_on_mask(mask)
        i, j = (location_pairs.location_a.contig_id, location_pairs.location_a.contig_id)
        np.add.at(self._inside_range_per_cell, (i, j), 1)
        distance_log_probs = self._distance_distribution.log_probability(dists[mask])
        np.add.at(self._distance_dist_weighted_counts, (i, j), distance_log_probs)


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
    px.imshow(connected_matrix, title='connected').show()
    px.imshow(disconnected_matrix, title='disconnected').show()
    optimal_squares = DirectOptimalSquares(connected_matrix, disconnected_matrix, sum(size_array), max_distance, max_splits=20)
    splits = optimal_squares.find_splits()
    return split_based_on_indices(path, splits)
