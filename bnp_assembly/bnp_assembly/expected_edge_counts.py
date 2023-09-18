from functools import lru_cache

import numpy as np

from bnp_assembly.plotting import px


class CumulativeDistribution:
    def __init__(self, intra_distribution: np.ndarray, p_noise: float = 0.3, genome_size: int = 1000000) -> object:

        proportion_covered_by_intra = (genome_size - len(intra_distribution)) / genome_size
        p_inside = 1 - p_noise * proportion_covered_by_intra
        self._distribution = intra_distribution / intra_distribution[-1] * p_inside
        self._noise_pmf = p_noise/genome_size
        self._genome_size = genome_size

    def __getitem__(self, x):
        return self.cdf(x)

    def cdf(self, x):
        assert x <= self._genome_size
        if x < len(self._distribution):
            return self._distribution[x]
        else:
            return self._distribution[-1] + self._noise_pmf * (x+1 - len(self._distribution))


class PairWithinContigDistribution:
    def __init__(self, genome_size, contig_start, contig_end, cumulative_distance_distribution):
        self._genome_size = genome_size
        self._contig_start = contig_start
        self._contig_end = contig_end
        self._cumulative_distance_distribution = cumulative_distance_distribution

    def log_probability(self, value):
        pass

    def sample(self, value):
        a = np.random.randint(genome_size)


class ExpectedEdgeCounts:
    def __init__(self, contig_dict, cumulative_distribution, max_distance=100000):
        self._contig_dict = contig_dict
        self._cumulative_distribution = cumulative_distribution
        self._genome_size = sum(contig_dict.values())
        self._max_distance = 100000

    @property
    @lru_cache()
    def _prob_pair_within_region_size(self):
        max_size = min(2*max(self._contig_dict.values()), self._genome_size)+1
        a = np.zeros(max_size)
        for i in range(max_size):
            a[i] = self._cumulative_distribution[i]
        px(name='splitting').line(a, title='real_cdf')
        probs = np.cumsum(a) / sum(self._contig_dict.values())
        assert np.all(probs) <= 1
        return np.insert(probs, 0, 0)

    def get_truncated_node_size(self, node_id):
        return min(self._contig_dict[node_id], self._max_distance)

    def get_expected_edge_count(self, edge):
        probs = self._prob_pair_within_region_size
        node_sizes = [self.get_truncated_node_size(node_side.node_id) for node_side in (edge.from_node_side, edge.to_node_side)]
        expected_on_both = probs[sum(node_sizes)]
        assert expected_on_both > 0, (expected_on_both, sum(node_sizes))
        expected_within_each = sum(probs[node_size] for node_size in node_sizes)
        assert expected_within_each < expected_on_both, (expected_on_both, expected_within_each)
        return (expected_on_both - expected_within_each) / 2
