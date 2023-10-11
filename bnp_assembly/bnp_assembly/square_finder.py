import logging
from itertools import chain
from typing import List
import plotly.express as px
import more_itertools
import numpy as np
import scipy.stats

from bnp_assembly.clustering import count_interactions
from bnp_assembly.contig_graph import ContigPath
import logging

from bnp_assembly.distance_distribution import distance_dist, CumulativeDist2d
from bnp_assembly.location import LocationPair

logger = logging.getLogger(__name__)


class OptimalSquares:
    def __init__(self, count_matrix, opportunity_matrix=1, opportunity_matrix_background=1, max_splits=20):
        self._count_matrix = count_matrix
        self._opportunity_matrix = np.broadcast_to(opportunity_matrix, self._count_matrix.shape)
        self._opportunity_matrix_background = np.broadcast_to(opportunity_matrix_background, self._count_matrix.shape)
        self._max_split = max_splits
        self._n_nodes = len(self._count_matrix)
        px.imshow(self._count_matrix, title='count').show()
        px.imshow(self._opportunity_matrix, title='opportunity').show()
        px.imshow(self._opportunity_matrix_background, title='opportunity background').show()
        px.imshow(self._count_matrix / self._opportunity_matrix, title='rate').show()

    def score_split(self, split_indices: List[int]) -> float:
        if len(split_indices) != len(set(split_indices)):
            return -np.inf

        inside_squares = []
        inside_opportunity = []
        outside_squares = []
        outside_opportunity = []
        for start, end in more_itertools.pairwise(split_indices):
            inside_triangle = self.get_inside_triangle(start, end, self._count_matrix)
            inside_squares.append(inside_triangle)
            inside_opportunity.append(self.get_inside_triangle(start, end, self._opportunity_matrix))
            outside_squares.append(self.get_outside_cells(start, end, self._count_matrix))
            outside_opportunity.append(self.get_outside_cells(start, end, self._opportunity_matrix))
            #self._opportunity_matrix_background[start:end, :start])
        log_likelihoods = [calculate_likelihoods(flatten(squares), flatten(opp)) for squares, opp in
                           [(inside_squares, inside_opportunity),
                            (outside_squares, outside_opportunity)]]
        s = sum(log_likelihoods)
        return s

    def get_outside_cells(self, start, end, matrix):
        return matrix[start:end, :start]

    def get_inside_triangle(self, start, end, matrix):
        inside_square = matrix[start:end, start:end]
        inside_triangle = inside_square[np.triu_indices(len(inside_square), k=0)]
        return inside_triangle

    def find_splits(self):
        max_splits = min(self._max_split, self._n_nodes - 1)
        splits = [0, self._n_nodes]
        cur_score = self.score_split(splits)
        for split_number in range(max_splits):
            logger.info(f'Finding split number {split_number}, cur score: {cur_score} cur_splits {splits}')
            scores = [(self.score_split(list(sorted(splits + [i]))), i) for i in range(1, self._n_nodes) if i not in splits]
            logger.info(f'Scores: {scores}, {max(scores)}')
            new_score, new_split = max(scores)
            if new_score <= cur_score:
                return splits
            cur_score = new_score
            splits = list(sorted(splits + [new_split]))
        return splits
        # return find_splits(self._count_matrix)


class DirectOptimalSquares(OptimalSquares):
    def __init__(self, connected_logprobs, disconnected_log_probs, max_splits=20):
        self._connected_logprobs = connected_logprobs
        self._disconnected_log_probs = disconnected_log_probs
        self._n_nodes = len(connected_logprobs)
        self._max_split = max_splits

    def score_split(self, split_indices: List[int]):
        scores = [self.get_inside_triangle(start, end, self._connected_logprobs) for start, end in more_itertools.pairwise(split_indices)]
        scores += [self.get_outside_cells(start, end, self._disconnected_log_probs) for start, end in more_itertools.pairwise(split_indices)]
        return sum(s.sum() for s in scores)

def flatten(squares):
    return np.concatenate([square.ravel() for square in squares])


def calculate_likelihoods(values, weights=1):
    weights = np.broadcast_to(weights, values.shape)
    base_rate = np.sum(values) / np.sum(weights)
    rates = base_rate * weights
    log_pmf = scipy.stats.poisson.logpmf(values, rates)
    assert np.all(np.isfinite(log_pmf)), (rates, values)
    np_sum = np.sum(log_pmf)
    assert np.isfinite(np_sum), (rates, values)
    return np_sum


# def _get_score_for_split(similarity_matrix: np.ndarray, split_indices: List[int]) -> float:
#     if len(split_indices) != len(set(split_indices)):
#         return -np.inf
#
#     intervals = more_itertools.pairwise(chain([0], split_indices, [len(similarity_matrix)]))
#     inside_squares = []
#     outside_squares = []
#     for start, end in intervals:
#         inside_square = similarity_matrix[start:end, start:end]
#         inside_squares.append(inside_square[np.triu_indices(len(inside_square), k=0)])
#         outside_squares.append(similarity_matrix[start:end, :start])
#     log_likelihoods = (calculate_likelihoods(flatten(squares)) for squares in (inside_squares, outside_squares))
#     return sum(log_likelihoods)
#
#
# def find_splits(similarity_matrix: np.ndarray, max_splits: int = 10) -> List[int]:
#     return OptimalSquares(similarity_matrix, max_splits=max_splits).find_splits()
#     # n_nodes = len(similarity_matrix)
#     # splits = [0, n_nodes]
#     # cur_score = get_score_for_split(similarity_matrix, splits)
#     # optimal_squares = OptimalSquares(similarity_matrix)
#
#     for split_number in range(max_splits):
#         logger.info(f'Finding split number {split_number}')
#         new_score, new_split = max(((optimal_squares.score_split(list(sorted(splits + [i]))), i)
#                                    for i in range(1, len(similarity_matrix))))
#         if new_score <= cur_score:
#             return splits
#         cur_score = new_score
#         splits = list(sorted(splits + [new_split]))


def split_based_on_indices(path: ContigPath, splits):
    edges = path.edges
    split_edges = [edges[i - 1] for i in splits[1:-1]]
    return path.split_on_edges(split_edges)


def get_opportunity_matrix(size_array, dist_2):
    all_splits = np.insert(np.cumsum(size_array), 0, 0)
    matrix = np.array([[dist_2.get_w_weight(abs(offset_a-offset_b), size_a, size_b) for (size_a, offset_a) in zip(size_array, all_splits)] for size_b, offset_b in zip(size_array, all_splits)])
    matrix[np.diag_indices_from(matrix)]*=2
    return np.array(matrix)


def squares_split(numeric_input_data, path: ContigPath):
    np.seterr(divide='raise')
    distance_distribution = distance_dist(next(numeric_input_data.location_pairs), numeric_input_data.contig_dict)
    # px.line(distance_distribution[::10]).show()
    dist_2 = CumulativeDist2d(distance_distribution)
    # px.line(dist_2._cumulative_cumulative_dist[::10]).show()
    interaction_counts = count_interactions(numeric_input_data.contig_dict, next(numeric_input_data.location_pairs))
    size_array = np.array(list(numeric_input_data.contig_dict.values()))
    opportunity_matrix = get_opportunity_matrix(size_array, dist_2)
    assert np.all(opportunity_matrix> 0), (opportunity_matrix, size_array, dist_2)
    opportunity_matrix_background = np.multiply.outer(size_array, size_array)
    optimal_squares = OptimalSquares(interaction_counts, opportunity_matrix, opportunity_matrix_background, max_splits=20)
    splits = optimal_squares.find_splits()
    return split_based_on_indices(path, splits)


class SpecificSquareEstimator:
    def __init__(self, starts_a, starts_b, ends_a, ends_bcontig_sizes):
        self._starts_a = starts_a[:, None]
        self._starts_b = starts_b[:, None]
        self._ends_a = ends_a[:, None]
        self._ends_b = ends_b[:, None]
        self._counter = np.zeros(len(starts_a), dtype=int)
        self._contig_sizes_mask = contig_sizes>=np.maximum(
            sizes_a+distance_a, sizes_b+distance_b)

    def register_location_pairs(self, location_pairs: LocationPair):
        mask = location_pairs.location_a.contig_id == location_pairs.location_b.contig_id
        location_pairs = location_pairs.subset_with_mask(mask)
        mask = (location_pairs.location_a.offset >= self._starts_a) & (location_pairs.location_a.offset < self._ends_a)
        mask &= (location_pairs.location_b.offset >= self._starts_b) & (location_pairs.location_b.offset < self._ends_b)
        self._counter += np.sum(mask, axis=1)


        for distance, sizes_a, sizes_b in zip(self._distances, self._sizes_a, self._sizes_b):
            mask = np.abs(location_pairs.location_a.offset - location_pairs.location_b.offset) == distance
            location_pairs = location_pairs.subset_with_mask(mask)
            mask = (location_pairs.location_a.offset < sizes_a) & (location_pairs.location_b.offset < sizes_b)
            location_pairs = location_pairs.subset_with_mask(mask)



