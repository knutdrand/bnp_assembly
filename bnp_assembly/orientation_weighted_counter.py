from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import scipy

from bnp_assembly.distance_distribution import DISTANCE_CUTOFF, distance_dist
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.forbes_score import DistanceDistribution
from bnp_assembly.graph_objects import NodeSide, Edge
from bnp_assembly.missing_data import find_missing_data_and_adjust
from bnp_assembly.orientation_distribution import OrientationDistribution
from bnp_assembly.plotting import px


class OrientationWeightedCounter:
    def __init__(self, contig_dict, read_pairs):
        self._contig_dict = contig_dict
        self._read_pairs = read_pairs
        self._distance_distribution = DistanceDistribution(self.__log_probs)
        self._distance_distribution.plot()
        self._distance_distribution.save('distance_distribution.npy')

    def get_distance_matrix(self, method='logprob'):
        assert method=='logprob', method
        pair_counts = self.calculate_log_prob_weighted_counts()
        distance_matrix = DirectedDistanceMatrix(len(self._contig_dict))
        for edge, value in pair_counts.items():
            distance_matrix[edge] = -np.log(value)
            distance_matrix[edge.reverse()] = -np.log(value)
        return distance_matrix

    def calculate_log_prob_weighted_counts(self):
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


    @property
    @lru_cache()
    def __point_probs(self):
        np.save('cumulative_distance_distribution.npy', self._cumulative_distance_distribution)
        base = np.diff(self._cumulative_distance_distribution)
        smoothed = scipy.ndimage.gaussian_filter1d(base, 10)
        # smoothed = smooth_sklearn(base)
        px(name='joining').line(smoothed, title='smoothed')
        '''
        px(name='joining').array(base[:100000], title='distribution')
        px(name='joining').line(base[:100000], title='distribution')

        px(name='joining').line(smoothed, title='smoothed')
        '''
        smoothed[-1] = 0
        for i in range(1, len(smoothed)//DISTANCE_CUTOFF+1):
            s = slice(i * DISTANCE_CUTOFF, (i + 1) * DISTANCE_CUTOFF)
            smoothed[s] = np.mean(smoothed[s])
        smoothed = smoothed + 0.000001 / len(smoothed)
        px(name='joining').line(smoothed / np.sum(smoothed), title='smoothed2')
        return smoothed / np.sum(smoothed)

    @property
    @lru_cache()
    def __log_probs(self):
        a = np.log(self.__point_probs)
        px(name='joining').array(a, title='log probs')
        return a

    def log_probability(self, distance):
        if distance >= len(self.__point_probs):
            distance = -1
        return self.__log_probs[distance]

    def __point_probability(self, distance):
        bin_size = 10
        i = distance // bin_size
        if i >= len(self.__point_probs):
            return 0
        return self.__point_probs[i] / bin_size

    def __probability_of_longer(self, distance: int) -> float:
        return 1 - self.__cumulative_distribution(distance)

    @property
    @lru_cache()
    def _cumulative_distance_distribution(self):
        return distance_dist(self._read_pairs, self._contig_dict)

    def __cumulative_distribution(self, distance):
        if distance >= len(self._cumulative_distance_distribution):
            return 1
        return self._cumulative_distance_distribution[distance]


class OrientationWeightedCountesWithMissing(OrientationWeightedCounter):
    def calculate_log_prob_weighted_counts(self):
        counts = super().calculate_log_prob_weighted_counts()
        adjusted_counts = find_missing_data_and_adjust(counts, self._contig_dict, self._read_pairs, self._cumulative_distance_distribution, 1000)
        return adjusted_counts

'''
class UnusedClass(OrientationWeightedCounter):
    @lru_cache()
    def __expected_pair_count(self, edge):
        return self._expected_node_side_counts[edge.from_node_side] * self._expected_node_side_counts[edge.to_node_side]

    def __calculate_observed_pair_count(self):
        pair_counts = Counter()
        for a, b in zip(self._read_pairs.location_a, self._read_pairs.location_b):
            right_weight_a = self._side_weights(a.offset, self._contig_dict[int(a.contig_id)])
            right_weight_b = self._side_weights(b.offset, self._contig_dict[int(b.contig_id)])
            raw_weights = {}
            for direction_a in ('l', 'r'):
                distance_a = a.offset if direction_a == 'l' else self._contig_dict[int(a.contig_id)] - a.offset - 1
                a_side = NodeSide(int(a.contig_id), direction_a)
                p_a = right_weight_a if direction_a == 'r' else 1 - right_weight_a
                for direction_b in ('l', 'r'):
                    b_side = NodeSide(int(b.contig_id), direction_b)
                    distance_b = b.offset if direction_b == 'l' else self._contig_dict[int(b.contig_id)] - b.offset - 1
                    total_dist = distance_a + distance_b
                    raw_weights[Edge(a_side, b_side)] = self.__point_probability(total_dist)
            T = sum(raw_weights.values())
            for edge, raw_weight in raw_weights.items():
                pair_counts[edge] += raw_weight / T
                pair_counts[edge.reverse()] += raw_weight / T
        DirectedDistanceMatrix.from_edge_dict(len(self._contig_dict), pair_counts).inversion_plot('counts')
        return pair_counts

    def __calculate_expected_node_side_counts(self):
        node_sides = [NodeSide(i, d) for i in range(len(self._contig_dict)) for d in ('l', 'r')]
        counter = {node_side: 0 for node_side in node_sides}
        for location in chain(self._read_pairs.location_a, self._read_pairs.location_b):
            right_side_prob = 0.5 * self.__probability_of_longer(
                self._contig_dict[int(location.contig_id)] - location.offset - 1)
            counter[NodeSide(location.contig_id, 'r')] += right_side_prob
            left_side_prob = 0.5 * self.__probability_of_longer(location.offset)
            counter[NodeSide(location.contig_id, 'l')] += left_side_prob
        return counter

    def __edge_distance(self, location: Location, direction):
        offset = location.offset
        node_id = int(location.contig_id)
        return offset if direction == 'l' else self._contig_dict[node_id] - offset - 1
'''
