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

logger = logging.getLogger(__name__)


class OptimalSquares:
    def __init__(self, count_matrix, opportunity_matrix=1, max_splits=20):
        self._rate_matrix = count_matrix/opportunity_matrix
        # self._cumulative_sum = np.cumsum(count_matrix, axis=0)
        # self._diagonal_sums = self._cumulative_sum.diagonal()-self._rate_matrix.diagonal()
        self._max_split = max_splits

    def score_split(self, split_indices: List[int]) -> float:
        similarity_matrix = self._rate_matrix

        if len(split_indices) != len(set(split_indices)):
            return -np.inf

        intervals = more_itertools.pairwise(chain([0], split_indices, [len(similarity_matrix)]))
        inside_squares = []
        outside_squares = []
        for start, end in intervals:
            inside_square = similarity_matrix[start:end, start:end]
            inside_squares.append(inside_square[np.triu_indices(len(inside_square), k=0)])
            outside_squares.append(similarity_matrix[start:end, :start])
        log_likelihoods = (calculate_likelihoods(flatten(squares)) for squares in (inside_squares, outside_squares))
        return sum(log_likelihoods)

    def find_splits(self):
        similarity_matrix = self._rate_matrix
        max_splits = self._max_split
        n_nodes = len(similarity_matrix)
        splits = [0, n_nodes]
        cur_score = self.score_split(splits)
        optimal_squares = OptimalSquares(similarity_matrix)

        for split_number in range(max_splits):
            logger.info(f'Finding split number {split_number}')
            new_score, new_split = max(((optimal_squares.score_split(list(sorted(splits + [i]))), i)
                                        for i in range(1, len(similarity_matrix))))
            if new_score <= cur_score:
                return splits
            cur_score = new_score
            splits = list(sorted(splits + [new_split]))

        return find_splits(self._count_matrix)


def flatten(squares):
    return np.concatenate([square.ravel() for square in squares])


def calculate_likelihoods(values, weights=1):
    weights = np.broadcast_to(weights, values.shape)
    base_rate = np.sum(values)/np.sum(weights)
    rates = base_rate * weights
    logpmf = scipy.stats.poisson.logpmf(values, rates)
    return np.sum(logpmf)


def _get_score_for_split(similarity_matrix: np.ndarray, split_indices: List[int]) -> float:
    if len(split_indices) != len(set(split_indices)):
        return -np.inf

    intervals = more_itertools.pairwise(chain([0], split_indices, [len(similarity_matrix)]))
    inside_squares = []
    outside_squares = []
    for start, end in intervals:
        inside_square = similarity_matrix[start:end, start:end]
        inside_squares.append(inside_square[np.triu_indices(len(inside_square), k=0)])
        outside_squares.append(similarity_matrix[start:end, :start])
    log_likelihoods = (calculate_likelihoods(flatten(squares)) for squares in (inside_squares, outside_squares))
    return sum(log_likelihoods)


def find_splits(similarity_matrix: np.ndarray, max_splits: int = 10) -> List[int]:
    return OptimalSquares(similarity_matrix, max_splits=max_splits).find_splits()
    # n_nodes = len(similarity_matrix)
    # splits = [0, n_nodes]
    # cur_score = get_score_for_split(similarity_matrix, splits)
    # optimal_squares = OptimalSquares(similarity_matrix)

    for split_number in range(max_splits):
        logger.info(f'Finding split number {split_number}')
        new_score, new_split = max(((optimal_squares.score_split(list(sorted(splits + [i]))), i)
                                   for i in range(1, len(similarity_matrix))))
        if new_score <= cur_score:
            return splits
        cur_score = new_score
        splits = list(sorted(splits + [new_split]))


def split_based_on_indices(path: ContigPath, splits):
    edges = path.edges
    split_edges = [edges[i-1] for i in splits[1:-1]]
    return path.split_on_edges(split_edges)


def squares_split(numeric_input_data, path: ContigPath):
    interaction_counts = count_interactions(numeric_input_data.contig_dict, next(numeric_input_data.location_pairs))
    size_array = np.array(list(numeric_input_data.contig_dict.values()))
    opportunity_matrix = np.multiply.outer(size_array, size_array)
    optimal_squares = OptimalSquares(interaction_counts, opportunity_matrix, max_splits=20)
    splits = optimal_squares.find_splits()
    return split_based_on_indices(path, splits)
