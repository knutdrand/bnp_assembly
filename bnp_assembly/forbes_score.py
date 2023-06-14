from functools import lru_cache
from itertools import chain, combinations, combinations_with_replacement, product

import numpy as np
import pandas as pd
import scipy

from .location import LocationPair, Location
from .graph_objects import NodeSide, Edge
from .distance_matrix import DirectedDistanceMatrix
from .distance_distribution import distance_dist
from collections import Counter, defaultdict
from scipy.stats import poisson

from .orientation_distribution import OrientationDistribution
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
    n_nodes = max(len(node_side_counts) // 2, 1)
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = max(sum(node_side_counts.values()), 1)
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


class DistanceDistribution:
    def __init__(self, log_probabilities):
        self._log_probabilities = log_probabilities

    def log_probability(self, distance):
        return self._log_probabilities[np.where(distance < len(self._log_probabilities), distance, -1)]

    @classmethod
    def load(cls, filename):
        return cls(np.load(filename))

    def save(self, filename):
        np.save(filename, self._log_probabilities)


class Forbes2:
    def __init__(self, contig_dict, read_pairs):
        self._contig_dict = contig_dict
        self._read_pairs = read_pairs
        self._expected_node_side_counts = self.calculate_expected_node_side_counts()
        self._side_weights = CumulativeSideWeight(self._cumulative_distance_distribution)
        self._distance_distribution = DistanceDistribution(self._log_probs)
        self._distance_distribution.save('distance_distribution.npy')

    def get_distance_matrix(self, method='logprob'):
        if method == 'logprob':
            pair_counts = self.calculate_log_prob_weighted_counts()
            node_side_counts = {node_side: 1 for node_side in self._expected_node_side_counts}
            return get_forbes_matrix(pair_counts, node_side_counts)
        else:
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
                    raw_weights[Edge(a_side, b_side)] = self.point_probability(total_dist)
            T = sum(raw_weights.values())
            for edge, raw_weight in raw_weights.items():
                pair_counts[edge] += raw_weight / T
                pair_counts[edge.reverse()] += raw_weight / T
                # p_b = right_weight_b if direction_b == 'r' else 1 - right_weight_b
                #    pair_counts[Edge(a_side, b_side)] += p_a * p_b
                #    pair_counts[Edge(b_side, a_side)] += p_a * p_b
        DirectedDistanceMatrix.from_edge_dict(len(self._contig_dict), pair_counts).inversion_plot('counts')
        return pair_counts

    def calculate_expected_node_side_counts(self):
        node_sides = [NodeSide(i, d) for i in range(len(self._contig_dict)) for d in ('l', 'r')]
        counter = {node_side: 0 for node_side in node_sides}
        for location in chain(self._read_pairs.location_a, self._read_pairs.location_b):
            right_side_prob = 0.5 * self.probability_of_longer(
                self._contig_dict[int(location.contig_id)] - location.offset - 1)
            counter[NodeSide(location.contig_id, 'r')] += right_side_prob
            left_side_prob = 0.5 * self.probability_of_longer(location.offset)
            counter[NodeSide(location.contig_id, 'l')] += left_side_prob
        return counter

    def edge_distance(self, location: Location, direction):
        offset = location.offset
        node_id = int(location.contig_id)
        return offset if direction == 'l' else self._contig_dict[node_id] - offset - 1

    def distribution_plot(self, i):
        m, n = self._contig_dict[i], self._contig_dict[i + 1]

    def calculate_log_prob_weighted_counts(self):
        node_pair_counts = Counter()
        edge_log_probs = Counter()
        positions = defaultdict(list)
        weights = Counter()
        scores = defaultdict(list)
        orientation_distributions = {(node_id_a, node_id_b): OrientationDistribution(self._contig_dict[int(node_id_a)],
                                                                                     self._contig_dict[int(node_id_b)],
                                                                                     self._distance_distribution) for
                                     node_id_a, node_id_b in product(self._contig_dict, repeat=2)}
        counts = Counter()
        for a, b in zip(self._read_pairs.location_a, self._read_pairs.location_b):
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

        self.plot_scores(positions, scores)
        table = pd.DataFrame([{'nodeid': node_id,
                               'p': counts[Edge(NodeSide(node_id, dir_a), NodeSide(node_id + 1, dir_b))],
                               'directions': f'{dir_a}->{dir_b}'}
                              for node_id in range(len(self._contig_dict) - 1) for dir_a, dir_b in
                              product('lr', repeat=2)])
        px(name='joining').bar(table, x='nodeid', y='p', color='directions', title='probs', barmode='group')
        return counts
        '''
        for a, b in zip(self._read_pairs.location_a, self._read_pairs.location_b):
            a_contig_id = int(a.contig_id)
            contig_id = int(b.contig_id)
            if a_contig_id < contig_id:
                positions[(a_contig_id, contig_id)].append((a.offset, b.offset))
            else:
                positions[(contig_id, a_contig_id)].append((b.offset, a.offset))
            distances = {}
            for direction_a in ('l', 'r'):
                distance_a = self.edge_distance(a, direction_a)
                a_side = NodeSide(int(a_contig_id), direction_a)
                for direction_b in ('l', 'r'):
                    b_side = NodeSide(int(contig_id), direction_b)
                    distance_b = self.edge_distance(b, direction_b)
                    # distance_b = b.offset if direction_b == 'l' else self._contig_dict[int(contig_id)] - b.offset - 1
                    total_dist = distance_a + distance_b
                    distances[Edge(a_side, b_side)] = total_dist

            node_pair_counts[(a_contig_id, contig_id)] += 1
            node_pair_counts[(contig_id, a_contig_id)] += 1
            log_probs = {edge: self.log_probability(total_dist) for edge, total_dist in distances.items()}
            # for edge, log_prob in log_probs.items():
            #    edge_log_probs[edge] += log_prob
            #    edge_log_probs[edge.reverse()] += log_prob
            tot_log_prob = scipy.special.logsumexp(list(log_probs.values()))
            probs = {edge: np.exp(log_prob - tot_log_prob)
                     for edge, log_prob in log_probs.items()}
            assert np.allclose(sum(probs.values()), 1), probs.values()
            for edge, prob in probs.items():
                weights[edge] += prob
                weights[edge.reverse()] += prob
            t = probs
            if a_contig_id < contig_id:
                scores[(a_contig_id, contig_id)].append(t[Edge(NodeSide(a_contig_id, 'r'), NodeSide(contig_id, 'l'))])
            else:
                scores[(contig_id, a_contig_id)].append(t[Edge(NodeSide(a_contig_id, 'l'), NodeSide(contig_id, 'r'))])
           '''

    def plot_scores(self, positions, scores):
        for i in range(len(self._contig_dict) - 1):
            p = positions[(i, i + 1)]
            s = scores[(i, i + 1)]
            if len(p):
                x, y = zip(*p)
                x = self._contig_dict[i] - np.array(x)
                px(name='joining').scatter(x=x, y=y, title=f'{i}-{i + 1}', color=s)

        '''
        weights = {}
        total_logprobs = defaultdict(int)
        print(node_pair_counts.keys())
        px(name='joining').bar(x=[f'{i}:{i + 1}' for i in range(len(self._contig_dict) - 1)],
                               y=[node_pair_counts[(i, i + 1)] for i in range(len(self._contig_dict) - 1)],
                               title='node pair counts')
        for (contig_a, contig_b), count in node_pair_counts.items():
            edges = [Edge(NodeSide(contig_a, dir_a), NodeSide(contig_b, dir_b))
                     for dir_a in ('l', 'r') for dir_b in ('l', 'r')]
            total_logprob = scipy.special.logsumexp([edge_log_probs[edge] for edge in edges])
            if contig_b == contig_a + 1:
                print(contig_a, contig_b, count, total_logprob, ([edge_log_probs[edge] for edge in edges]))
            total_logprobs[(contig_a, contig_b)] = total_logprob
            for edge in edges:
                weights[edge] = count * np.exp(edge_log_probs[edge] - total_logprob)

        t = [[i, edge_log_probs[Edge(NodeSide(i, dir_a), NodeSide(i + 1, dir_b))], total_logprobs[(i, i + 1)],
              (dir_a, dir_b)]
             for i in range(len(self._contig_dict) - 1) for dir_a in ('l', 'r') for dir_b in ('l', 'r')]
        table = pd.DataFrame(t, columns=['nodeid', 'logprob', 'total_logprob', 'directions'])
        table['p'] = np.exp(table['logprob'] - table['total_logprob'])
        print(table.groupby('directions').mean())
        px(name='joining').histogram(table, x='logprob', color='directions', barmode='group')
        px(name='joining').line(table, x='nodeid', y='logprob', color='directions', line_shape='hv', title='logprobs')
        px(name='joining').bar(table, x='nodeid', y='p', color='directions', title='probs', )
        DirectedDistanceMatrix.from_edge_dict(len(self._contig_dict), weights).inversion_plot('counts')
        return weights
        '''

    @property
    @lru_cache()
    def _point_probs(self):
        base = np.diff(self._cumulative_distance_distribution)
        smoothed = scipy.ndimage.gaussian_filter1d(base, 10)
        px(name='joining').line(smoothed, title='smoothed')
        '''
        px(name='joining').array(base[:100000], title='distribution')
        px(name='joining').line(base[:100000], title='distribution')

        px(name='joining').line(smoothed, title='smoothed')
        '''
        smoothed[-1] = 0
        smoothed = smoothed + 0.000001 / len(smoothed)
        px(name='joining').line(smoothed / np.sum(smoothed), title='smoothed2')
        return smoothed / np.sum(smoothed)

    @property
    @lru_cache()
    def _log_probs(self):
        a = np.log(self._point_probs)
        px(name='joining').array(a, title='log probs')
        return a

    def log_probability(self, distance):
        if distance >= len(self._point_probs):
            distance = -1
        return self._log_probs[distance]

    def point_probability(self, distance):
        bin_size = 10
        i = distance // bin_size
        if i >= len(self._point_probs):
            return 0
        return self._point_probs[i] / bin_size

    def probability_of_longer(self, distance: int) -> float:
        return 1 - self.cumulative_distribution(distance)

    @property
    @lru_cache()
    def _cumulative_distance_distribution(self):
        return distance_dist(self._read_pairs, self._contig_dict)

    def cumulative_distribution(self, distance):
        if distance >= len(self._cumulative_distance_distribution):
            return 1
        return self._cumulative_distance_distribution[distance]


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
