from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import scipy

from bnp_assembly.distance_distribution import DISTANCE_CUTOFF, distance_dist, DistanceDistribution
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_scorer import EdgeScorer
from bnp_assembly.graph_objects import NodeSide, Edge
from bnp_assembly.missing_data import find_missing_data_and_adjust
from bnp_assembly.orientation_distribution import OrientationDistribution
from bnp_assembly.plotting import px


class OrientationWeightedCounter(EdgeScorer):
    def __init__(self, contig_dict, read_pairs, cumulative_length_distribution=None):
        self._contig_dict = contig_dict
        self._read_pairs = read_pairs
        if cumulative_length_distribution is None:
            cumulative_length_distribution = distance_dist(read_pairs, contig_dict)
        self._cumulative_distance_distribution = cumulative_length_distribution
        self._distance_distribution = DistanceDistribution.from_cumulative_distribution(self._cumulative_distance_distribution)
        self._distance_distribution.plot()

    def get_distance_matrix(self, method='logprob'):
        assert method=='logprob', method
        pair_counts = self._calculate_log_prob_weighted_counts()
        distance_matrix = DirectedDistanceMatrix(len(self._contig_dict))
        for edge, value in pair_counts.items():
            distance_matrix[edge] = -np.log(value)
            distance_matrix[edge.reverse()] = -np.log(value)
        return distance_matrix

    def _calculate_log_prob_weighted_counts(self):
        cutoff_distance = DISTANCE_CUTOFF
        positions = defaultdict(list)
        scores = defaultdict(list)
        orientation_distributions = {(node_id_a, node_id_b): OrientationDistribution(self._contig_dict[int(node_id_a)],
                                                                                     self._contig_dict[int(node_id_b)],
                                                                                     self._distance_distribution) for
                                     node_id_a, node_id_b in product(self._contig_dict, repeat=2)}
        counts = Counter()
        for a, b in zip(self._read_pairs.location_a, self._read_pairs.location_b):
            if a.offset > cutoff_distance and a.offset<self._contig_dict[int(a.contig_id)]-cutoff_distance:
                continue
            if b.offset > cutoff_distance and b.offset<self._contig_dict[int(b.contig_id)]-cutoff_distance:
                continue
            pair = (int(a.contig_id), int(b.contig_id))
            orientation_distribution = orientation_distributions[pair]
            probability_dict = orientation_distribution.orientation_distribution(a.offset, b.offset)
            for (dir_a, dir_b), probability in probability_dict.items():
                edge = Edge(NodeSide(int(a.contig_id), dir_a), NodeSide(int(b.contig_id), dir_b))
                counts[edge] += probability
                counts[edge.reverse()] += probability
            if a.contig_id < b.contig_id:
                positions[pair].append((a.offset, b.offset))
                scores[pair].append(probability_dict[('r', 'l')])
            else:
                positions[pair[::-1]].append((b.offset, a.offset))
                scores[pair[::-1]].append(probability_dict[('l', 'r')])
        self.positions = positions
        self.scores = scores
        self.plot_scores(positions, scores)
        table = pd.DataFrame([{'nodeid': node_id,
                               'p': counts[Edge(NodeSide(node_id, dir_a), NodeSide(node_id + 1, dir_b))],
                               'directions': f'{dir_a}->{dir_b}'}
                              for node_id in range(len(self._contig_dict) - 1) for dir_a, dir_b in
                              product('lr', repeat=2)])
        px(name='joining').bar(table, x='nodeid', y='p', color='directions', title='probs', barmode='group')
        return counts

    def plot_scores(self, positions, scores, edges=None):
        if edges is None:
            pairs = [(i, i + 1) for i in range(len(self._contig_dict) - 1)]
        else:
            pairs = [(edge.from_node_side.node_id, edge.to_node_side.node_id) for edge in edges]
        for i, j in pairs:
            i, j = min(i, j), max(i, j)
            p = positions[(i, j)]
            s = scores[(i, j)]
            print(i, j, len(p))
            if len(p):
                x, y = zip(*p)
                x = self._contig_dict[i] - np.array(x)
                px(name='joining').scatter(x=x, y=y, title=f'{i}to{j}', color=s)

    def __cumulative_distribution(self, distance):
        if distance >= len(self._cumulative_distance_distribution):
            return 1
        return self._cumulative_distance_distribution[distance]


class OrientationWeightedCountesWithMissing(OrientationWeightedCounter):
    def _calculate_log_prob_weighted_counts(self):
        counts = super()._calculate_log_prob_weighted_counts()
        adjusted_counts = find_missing_data_and_adjust(counts, self._contig_dict, self._read_pairs, self._cumulative_distance_distribution, 1000)
        return adjusted_counts
