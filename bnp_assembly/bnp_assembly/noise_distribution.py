from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import logsumexp

from bnp_assembly.graph_objects import Edge, NodeSide
from bnp_assembly.plotting import px


class NoiseDistribution:

    def __init__(self, contig_dict, distance_matrix, contig_path, max_distance=100000):
        self._contig_dict = contig_dict
        self._distance_matrix = distance_matrix
        self._contig_path = contig_path
        self.distance_cutoff  = max_distance
        values = [self.transform(distance_matrix[edge]) / self.size_factor(edge) for edge in
                  [Edge(NodeSide(i, dir_a), NodeSide(j, dir_b)) for i  in contig_dict.keys() for j in contig_dict.keys()
                   for
                   dir_a in 'lr' for dir_b in 'lr' if i != j]]
        print(values)
        px(name='splitting').histogram(values, title='all_edge_scores')


    def transform(self, weight):
        return np.exp(-weight)

    @lru_cache()
    def _dist(self, x):
        return scipy.stats.poisson(mu=np.mean(self.get_non_neighbour_scores())).logpmf(x)

    def rate(self, edge):
        return self._mean_rate() * self.size_factor(edge)

    def edge_probability(self, edge):
        edge_scores = self.get_non_neighbour_scores()
        edge_scores = edge_scores[:int(len(edge_scores) * 0.9)]
        distribution = scipy.stats.poisson(mu=self.size_factor(edge) * edge_scores)
        f = lambda x: logsumexp(distribution.logpmf(x), ) - np.log(len(edge_scores))
        g = lambda x: scipy.stats.poisson(mu=self._mean_rate() * self.size_factor(edge)).logpmf(x)
        x = np.arange(100)
        count = np.int64(self.transform(self._distance_matrix[edge]))
        logpmfs = distribution.logpmf(count)
        assert len(logpmfs) == len(edge_scores)

        r = logsumexp(logpmfs) - np.log(len(logpmfs))
        if False:
            # Should be lazy df construction
            df = [(x, np.exp(f(i)), 'new') for i in x]
            df.extend([(x, np.exp(g(i)), 'old') for i in x])
            df = pd.DataFrame(df, columns=['x', 'y', 'type'])
            px(name='splitting').line(df, x='x', y='y', color='type', title=f'dist {edge}-{count}')
        # px(name='splitting').line([np.exp(f(i)) for i in x], title=f'dist {edge}')
        # px(name='splitting').line([np.exp(g(i)) for i in x], title=f'dist-old {edge}')
        # old_logpmf = scipy.stats.poisson(mu=self._mean_rate() * self.size_factor(edge)).logpmf(count)
        # print(count, np.exp(r), np.exp(old_logpmf))
        return r
        return scipy.stats.poisson(mu=self._mean_rate() * self.size_factor(edge)).logpmf(
            np.int64(self.transform(self._distance_matrix[edge])))

    @lru_cache()
    def _mean_rate(self):
        return np.mean(self.get_non_neighbour_scores())

    def log_probability(self, count):
        return self._dist(count)
        # fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)

    def cdf(self, score):
        return np.searchsorted(self.get_non_neighbour_scores(), score) / len(self.get_non_neighbour_scores())

    @lru_cache()
    def get_neighbour_node_ids(self):
        neighbours = {(edge.from_node_side.node_id, edge.to_node_side.node_id) for edge in self._contig_path.edges}
        neighbours |= {(edge.to_node_side.node_id, edge.from_node_side.node_id) for edge in self._contig_path.edges}
        neighbours |= {(int(node_id), int(node_id)) for node_id in self._contig_dict.keys()}
        return neighbours

    def get_all_possible_edges(self):
        return (Edge(NodeSide(na, da), NodeSide(nb, db)) for (da, na, db, nb) in
                product('lr', self._contig_dict.keys(), repeat=2))

    def get_non_neighbour_edges(self):
        return (edge for edge in self.get_all_possible_edges() if
                (edge.from_node_side.node_id, edge.to_node_side.node_id) not in self.get_neighbour_node_ids())

    @lru_cache()
    def get_non_neighbour_scores(self):
        counts = [self.transform(self._distance_matrix[edge]) for edge in self.get_non_neighbour_edges()]
        size_factors = [self.size_factor(edge) for edge in self.get_non_neighbour_edges()]

        px(name='splitting').scatter(x=size_factors, y=counts, title='size_factors')

        scores = np.sort(np.array(counts) / np.array(size_factors))
        px(name='splitting').histogram(x=scores, title='noise_scores')
        return scores

        # return np.sort([np.exp(-self._distance_matrix[edge]) / self.size_factor(edge) for edge in self.get_non_neighbour_edges()])

    def truncated_node_size(self, node_id):
        return min(self._contig_dict[node_id], self.distance_cutoff * 2)

    def size_factor(self, edge):
        return self.truncated_node_size(edge.from_node_side.node_id) * self.truncated_node_size(
            edge.to_node_side.node_id)
