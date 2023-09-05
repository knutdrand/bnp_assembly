from collections import Counter, defaultdict
from functools import singledispatchmethod
from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from bnp_assembly.contig_sizes import ContigSizes
from bnp_assembly.distance_distribution import DISTANCE_CUTOFF, distance_dist, DistanceDistribution
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_counts import EdgeCounts
from bnp_assembly.edge_scorer import EdgeScorer
from bnp_assembly.graph_objects import NodeSide, Edge
from bnp_assembly.location import LocationPair
from bnp_assembly.missing_data import find_missing_data_and_adjust
from bnp_assembly.orientation_distribution import OrientationDistribution
from bnp_assembly.plotting import px


class OrientationDistributions:
    def __init__(self, contig_sizes: ContigSizes, distance_distribution: DistanceDistribution):
        self._contig_sizes = contig_sizes
        self._distance_distribution = distance_distribution
        self._orientation_distributions = {
            (node_id_a, node_id_b): OrientationDistribution(self._contig_sizes[node_id_a],
                                                            self._contig_sizes[node_id_b],
                                                            self._distance_distribution) for
            node_id_a, node_id_b in product(self._contig_sizes, repeat=2)}

    @singledispatchmethod
    def __getitem__(self, pair):
        raise ValueError(f"Invalid index for {self.__class__.__name__}: {pair}")

    @__getitem__.register
    def _(self, pair: tuple):
        return self._orientation_distributions[pair]

    @__getitem__.register
    def _(self, pair: LocationPair):
        node_ids = (int(pair.location_a.contig_id), int(pair.location_b.contig_id))
        offsets = (pair.location_a.offset, pair.location_b.offset)
        return self._orientation_distributions[node_ids].distribution_matrix(*offsets)


def create_distance_matrix(n_nodes, pair_counts):
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    for edge, value in pair_counts.items():
        distance_matrix[edge] = -np.log(value)
        distance_matrix[edge.reverse()] = -np.log(value)
    return distance_matrix


class OrientationWeightedCounter(EdgeScorer):

    def __init__(self, contig_dict, read_pairs=None, cumulative_length_distribution=None):
        self._contig_dict = ContigSizes.from_dict(contig_dict)
        self._read_pairs = read_pairs
        if cumulative_length_distribution is None:
            cumulative_length_distribution = distance_dist(read_pairs, contig_dict)
        self._cumulative_distance_distribution = cumulative_length_distribution
        self._distance_distribution = DistanceDistribution.from_cumulative_distribution(
            self._cumulative_distance_distribution)
        self._distance_distribution.plot()
        self.orientation_distributions = OrientationDistributions(self._contig_dict, self._distance_distribution)
        self.__orientation_distributions = {
            (node_id_a, node_id_b): OrientationDistribution(self._contig_dict[int(node_id_a)],
                                                            self._contig_dict[int(node_id_b)],
                                                            self._distance_distribution) for
            node_id_a, node_id_b in product(self._contig_dict, repeat=2)}
        self._counts = EdgeCounts(len(self._contig_dict))
        self.positions = defaultdict(list)
        self.scores = defaultdict(list)

    def get_distance_matrix(self, method='logprob'):
        assert method == 'logprob', method
        self._calculate_log_prob_weighted_counts()
        return self.finalize()

    def finalize(self):
        pair_counts = self._counts
        n_nodes = len(self._contig_dict)
        return create_distance_matrix(n_nodes, pair_counts)

    def _calculate_log_prob_weighted_counts(self):
        if isinstance(self._read_pairs, LocationPair):
            read_pair_stram = [self._read_pairs]
        else:
            read_pair_stram = self._read_pairs
        for read_pair in read_pair_stram:
            self.register_location_pairs(read_pair)
            #for a, b in zip(read_pair.location_a, read_pair.location_b):
            #    self.register_location_pair(LocationPair(a, b))
            #    self._register_for_plots(LocationPair(a, b))

        self.plot_scores(self.positions, self.scores)
        table = pd.DataFrame([{'nodeid': node_id,
                               'p': self._counts[Edge(NodeSide(node_id, dir_a), NodeSide(node_id + 1, dir_b))],
                               'directions': f'{dir_a}->{dir_b}'}
                              for node_id in range(len(self._contig_dict) - 1) for dir_a, dir_b in
                              product('lr', repeat=2)])
        px(name='joining').bar(table, x='nodeid', y='p', color='directions', title='probs', barmode='group')
        return self._counts

    def _register_for_plots(self, location_pair):
        a, b = location_pair.location_a, location_pair.location_b
        pair = (int(a.contig_id), int(b.contig_id))
        orientation_distribution = self.orientation_distributions[pair]
        # probabilities = orientation_distribution[location_pair]# (a.offset, b.offset)
        probability_dict = orientation_distribution.orientation_distribution(a.offset, b.offset)
        if a.contig_id < b.contig_id:
            self.positions[pair].append((a.offset, b.offset))
            self.scores[pair].append(probability_dict[('r', 'l')])
        else:
            self.positions[pair[::-1]].append((b.offset, a.offset))
            self.scores[pair[::-1]].append(probability_dict[('l', 'r')])

    def register_location_pairs(self, location_pairs):
        for a, b in zip(location_pairs.location_a, location_pairs.location_b):
            self.register_location_pair(LocationPair(a, b))
            self._register_for_plots(LocationPair(a, b))

    def register_location_pair(self, location_pair):
        a, b = location_pair.location_a, location_pair.location_b
        if DISTANCE_CUTOFF < a.offset < self._contig_dict[int(a.contig_id)] - DISTANCE_CUTOFF:
            return
        if DISTANCE_CUTOFF < b.offset < self._contig_dict[int(b.contig_id)] - DISTANCE_CUTOFF:
            return
        probability_matrix = self.orientation_distributions[location_pair]
        pair = (int(a.contig_id), int(b.contig_id))
        self._counts[pair] += probability_matrix
        self._counts[pair[::-1]] += probability_matrix.T# [::-1, ::-1]
        '''
        for (dir_a, dir_b), probability in probability_dict.items():
            edge = Edge(NodeSide(int(a.contig_id), dir_a),
                        NodeSide(int(b.contig_id), dir_b))
            self._counts[edge] += probability
            self._counts[edge.reverse()] += probability
        '''

    def plot_scores(self, positions, scores, edges=None):
        if edges is None:
            pairs = [(i, i + 1) for i in range(len(self._contig_dict) - 1)]
        else:
            pairs = [(edge.from_node_side.node_id, edge.to_node_side.node_id) for edge in edges]
        for i, j in pairs:
            i, j = min(i, j), max(i, j)
            p = positions[(i, j)]
            s = scores[(i, j)]
            if len(p):
                x, y = zip(*p)
                x = self._contig_dict[i] - np.array(x)
                px(name='joining').scatter(x=x, y=y, title=f'{i}to{j}', color=s)

    @property
    def counts(self):
        return self._counts


def add_dict_counts(dict_a, dict_b):
    return {key: dict_a.get(key, 0) + dict_b.get(key, 0) for key in set(dict_a) | set(dict_b)}


class OrientationWeightedCountesWithMissing(OrientationWeightedCounter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bin_counts_for_missing = Counter()

    def _calculate_log_prob_weighted_counts(self):
        super()._calculate_log_prob_weighted_counts()
        adjusted_counts = find_missing_data_and_adjust(self._counts, self._contig_dict, self._read_pairs,
                                                       self._cumulative_distance_distribution, 1000)
        self._counts = adjusted_counts
        return adjusted_counts

    def __register_location_pair(self, location_pair):
        super().register_location_pair(location_pair)
        self._register_bincount(location_pair)

    def _register_bincount(self, location_pair):
        counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, read_pairs)
