from functools import lru_cache
from itertools import chain

import numpy as np
from .location import LocationPair, Location
from .graph_objects import NodeSide, Edge
from .distance_matrix import DirectedDistanceMatrix
from .distance_distribution import calculate_distance_distritbution, distance_dist
from collections import Counter, defaultdict
from scipy.stats import poisson
from .plotting import px
import typing as tp


def get_chromosome_end_probabilities(contig_dict: tp.Dict[str, int], node_side_counts: tp.Dict[NodeSide, int]):
    poisson_regression_model = sklearn.linear_model.PoissonRegressor()
    X = np.array([contig_dict[node_side.node_id] for node_side in node_side_counts])[:, None]
    y = np.array([node_side_counts[node_side] for node_side in node_side_counts])
    poisson_regression_model.fit(X, y)
    rate = poisson_regression_model.predict(X)
    p = poisson.pmf(y, rate)

    return {node_side: poisson(node_side_counts[node_side]).pmf(contig_dict[node_side.node_id])
            for node_side in node_side_counts}


def get_pair_counts(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs):
    F = distance_dist(location_pairs, contig_dict)
    px(name='joining').line(F, title='distance')
    # calculate_distance_distritbution(list(contig_dict.values()),
    # [np.abs(a.offset-b.offset)
    #                                      for a, b in zip(location_pairs.location_a, location_pairs.location_b)
    #                                      if a.contig_id == b.contig_id])
    return count_window_combinastions(contig_dict, location_pairs, CumulativeSideWeight(F))


def get_node_side_counts(pair_counts):
    node_sides = {edge.from_node_side for edge in pair_counts} | {edge.to_node_side for edge in pair_counts}
    return {node_side: sum(
        (pair_counts[Edge(node_side, node_side2)] + pair_counts[Edge(node_side2, node_side)]) / 2 for node_side2 in
        node_sides if node_side2.node_id != node_side.node_id) for node_side in node_sides}
    # return {node_side: sum(value for key, value in pair_counts.items() if key.from_node_side == node_side) for node_side in node_sides}


def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs):
    pair_counts = get_pair_counts(contig_dict, location_pairs)
    node_side_counts = get_node_side_counts(pair_counts)
    return get_forbes_matrix(pair_counts, node_side_counts)


def get_rate(count_a, count_b, N, alpha=1):
    return (count_a + alpha) * (count_b + alpha) / N


def get_pscore_matrix(pair_counts: tp.Dict[Edge, float], node_side_counts: tp.Dict[NodeSide, float], alpha=1):
    n_nodes = len(node_side_counts) // 2
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = sum(node_side_counts.values())  # Should this be divided by two?
    for edge, value in pair_counts.items():
        rate = get_rate(node_side_counts[edge.from_node_side], node_side_counts[edge.to_node_side], N, 0)
        # node_side_counts[edge.from_node_side]*node_side_counts[edge.to_node_side]/N
        p_score = poisson.sf(pair_counts[edge] - 1, rate)
        distance_matrix[edge] = p_score
        distance_matrix[edge.reverse()] = p_score
    return distance_matrix


def get_forbes_matrix(pair_counts, node_side_counts, alpha=1):
    n_nodes = len(node_side_counts) // 2
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = sum(node_side_counts.values())
    for edge, count in pair_counts.items():
        rate = get_rate(node_side_counts[edge.from_node_side], node_side_counts[edge.to_node_side], N, alpha)
        score = (pair_counts[edge] + (alpha / (n_nodes * 2))) / rate
        # score = N*(pair_counts[edge]+alpha/(n_nodes*4))/((node_side_counts[edge.from_node_side]+alpha)*(node_side_counts[edge.to_node_side])+alpha)
        distance_matrix[edge] = -np.log(score)
        distance_matrix[edge.reverse()] = -np.log(score)
    return distance_matrix


def _naive_side_weight(position, size):
    return position / size


class CumulativeSideWeight:
    def __init__(self, cumulative_distribution):
        self.F = cumulative_distribution

    def __call__(self, position, size):
        '''
        P(s='R'| other=outside) = P(other=outside|s='r')P(s='r')/P(other=outside)
        P(s='r') = 0.5
        P(other=outside|s='r') = 1-F(size-position-1)
        P(other=outside|s='l') = 1-F(position)
        '''
        wR = (1 - self.F[size - position - 1])
        wL = (1 - self.F[position])
        return wR / (wR + wL)


class Forbes2:
    def __init__(self, contig_dict, read_pairs):
        self._contig_dict = contig_dict
        self._read_pairs = read_pairs
        self._expected_node_side_counts = self.calculate_expected_node_side_counts()
        self._side_weights = CumulativeSideWeight(self._distance_distribution)

    def get_distance_matrix(self):
        pair_counts = self.calculate_observed_pair_count()
        return get_forbes_matrix(pair_counts, self._expected_node_side_counts)

    @lru_cache()
    def expected_pair_count(self, edge):
        return self._expected_node_side_counts[edge.from_node_side] * self._expected_node_side_counts[edge.to_node_side]

    def calculate_observed_pair_count(self):
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
                    raw_weights[Edge(a_side, b_side)] = self.point_probs(total_dist)
            T = sum(raw_weights.values())
            for edge, raw_weight in raw_weights.items():
                pair_counts[edge] += raw_weight / T
                pair_counts[edge.reverse()] += raw_weight / T
                #p_b = right_weight_b if direction_b == 'r' else 1 - right_weight_b
                #    pair_counts[Edge(a_side, b_side)] += p_a * p_b
                #    pair_counts[Edge(b_side, a_side)] += p_a * p_b
        DirectedDistanceMatrix.from_edge_dict(len(self._contig_dict), pair_counts).inversion_plot('counts')
        return pair_counts

    def calculate_expected_node_side_counts(self):
        node_sides = [NodeSide(i, d) for i in range(len(self._contig_dict)) for d in ('l', 'r')]
        counter = {node_side: 0 for node_side in node_sides}
        for location in chain(self._read_pairs.location_a, self._read_pairs.location_b):
            right_side_prob = 0.5 * self.probability_of_longer(self._contig_dict[int(location.contig_id)] - location.offset-1)
            counter[NodeSide(location.contig_id, 'r')] += right_side_prob
            left_side_prob = 0.5 * self.probability_of_longer(location.offset)
            counter[NodeSide(location.contig_id, 'l')] += left_side_prob
        return counter

    @property
    @lru_cache()
    def _point_probs(self):
        bin_size = 10
        base =  np.diff(self._distance_distribution[::bin_size])
        base= base+0.000001
        return base/np.sum(base)


    def point_probs(self, distance):
        bin_size = 10
        i = distance//bin_size
        if i >= len(self._point_probs):
            return 0
        return self._point_probs[i]/bin_size

    def probability_of_longer(self, distance: int) -> float:
        return 1 - self.cumulative_distribution(distance)

    @property
    @lru_cache()
    def _distance_distribution(self):
        return distance_dist(self._read_pairs, self._contig_dict)

    def cumulative_distribution(self, distance):
        if distance >= len(self._distance_distribution):
            return 1
        return self._distance_distribution[distance]


def count_window_combinastions(contig_dict: tp.Dict[str, int], location_pairs: LocationPair,
                               side_weight_func=_naive_side_weight) -> tp.Tuple[Counter, Counter]:
    node_sides = [NodeSide(i, d) for i in range(len(contig_dict)) for d in ('l', 'r')]
    pair_counts = {Edge(a, b): 0 for a in node_sides for b in node_sides}  # if not a.node_id==b.node_id}
    tmp = []
    out_locations = defaultdict(list)
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        # if a.contig_id == b.contig_id:
        #    continue
        # if a.contig_id != b.contig_id:
        #    out_locations[int(a.contig_id)].append(a.offset)
        #    out_locations[int(b.contig_id)].append(b.offset)
        right_weight_a = side_weight_func(a.offset, contig_dict[int(a.contig_id)])
        right_weight_b = side_weight_func(b.offset, contig_dict[int(b.contig_id)])
        for direction_a in ('l', 'r'):
            a_side = NodeSide(int(a.contig_id), direction_a)
            p_a = right_weight_a if direction_a == 'r' else 1 - right_weight_a
            for direction_b in ('l', 'r'):
                b_side = NodeSide(int(b.contig_id), direction_b)
                p_b = right_weight_b if direction_b == 'r' else 1 - right_weight_b
                pair_counts[Edge(a_side, b_side)] += p_a * p_b
                pair_counts[Edge(b_side, a_side)] += p_a * p_b
    for name, locations in out_locations.items():
        px(name='joining').histogram(locations, title=str(name), nbins=contig_dict[name] // 1000)
    return pair_counts
