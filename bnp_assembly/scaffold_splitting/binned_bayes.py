from collections import defaultdict

import numpy as np
from functools import lru_cache

import pandas as pd
import plotly.express as _px
import typing as tp
from ..plotting import px as px_func
import scipy

px = px_func(name='splitting')

from ..interaction_matrix import InteractionMatrix

class Yahs:
    def __init__(self, count_matrix: np.ndarray,
                 contig_start_stops: tp.Dict[str, tp.Tuple[tp.Any, tp.Tuple[int, int]]]):
        self._count_matrix = count_matrix
        self._contig_start_stops = contig_start_stops
        n_bins = len(self._count_matrix)
        assert all(start < n_bins and stop <= n_bins for start, stop in self._contig_start_stops.values()), (
            n_bins, self._contig_start_stops)

    def save(self, filename: str):
        np.savez(filename, count_matrix=self._count_matrix, contig_ids=np.array(list(self._contig_start_stops.keys())),
                 start_stops = np.array(list(self._contig_start_stops.values())))

    @classmethod
    def load(cls, filename):
        with np.load(filename) as f:
            return cls(f['count_matrix'], dict(zip(f['contig_ids'], f['start_stops'])))

    def plot(self):
        pass

    def contig_matrix(self, n):
        start, stop = self._contig_start_stops[n]
        return self._count_matrix[start:stop, start:stop]

    def inter_matrix(self, m, n):
        start_m, stop_m = self._contig_start_stops[m]
        start_n, stop_n = self._contig_start_stops[n]
        sub_matrix = self._count_matrix[start_m:stop_m, start_n:stop_n]
        assert sub_matrix.shape == (self.b(m), self.b(n)), (sub_matrix.shape, self.b(m), self.b(n))
        return sub_matrix

    @lru_cache()
    def b(self, m):
        start, stop = self._contig_start_stops[m]
        assert stop - start > 0, (start, stop)
        return stop - start

    @lru_cache()
    def B(self, m, n, k):
        assert n == m + 1
        return sum(1
                   for i in range(max(self.b(m) - k, 0), self.b(m))
                   for j in range(min(k, self.b(n)))
                   if self.delta(m, n, i, j) == k)

    @lru_cache()
    def delta(self, *args):
        if len(args) == 2:
            i, j = args
            return np.abs(i - j)
        elif len(args) == 3:
            _, i, j = args
            return np.abs(i - j)
        elif len(args) == 4:
            m, n, i, j = args
            assert n == m + 1
            return self.delta(i, self.b(m) + j)

    @lru_cache()
    def c(self, *args):
        if len(args) == 2:
            i, j = args
            assert i >= 0 and j >= 0
            return self._count_matrix[i, j]
        elif len(args) == 3:
            n, i, j = args
            assert i >= 0 and j >= 0
            return self.contig_matrix(n)[i, j]
        elif len(args) == 4:
            m, n, i, j = args
            assert n == m + 1
            assert i >= 0 and j >= 0
            assert i < self.b(m) and j < self.b(n), (i, j, self.b(m), self.b(n))
            return self.inter_matrix(m, n)[i, j]

    @lru_cache()
    def w(self, *args):
        return self.c(*args) / self.E(self.delta(*args))

    @property
    def n_contigs(self):
        return len(self._contig_start_stops)

    @lru_cache()
    def E(self, d):
        # assert 0 < d <= self.D
        values = [self.c(n, i, j)
                  for n in range(self.n_contigs)
                  for i in range(min(d + 1, self.b(n))) for j in
                  range(min(d + 1, self.b(n))) if self.delta(n, i, j) == d]
        result = np.median(values)
        # assert result > 0
        return result

    @lru_cache()
    def f(self, m, n, d):
        if d <= 1:
            return 1
        weights = [self.w(m, n, i, j)
                   for i in range(max(self.b(m) - d, 0), self.b(m))
                   for j in range(0, min(d + 1, self.b(n)))
                   if 1 <= self.delta(m, n, i, j) < d]
        numerator = sum(weights)
        denominator = len(weights)

        assert denominator > 0, (m, n, d, numerator)
        return numerator / denominator

    @lru_cache()
    def s(self, m, n):
        scores = [self.f(m, n, self.delta(m, n, i, j)) * self.w(m, n, i, j)
                  for i in range(max(self.b(m) - self.D - 1, 0), self.b(m))
                  for j in range(min(self.D + 1, self.b(n)))
                  if 1 <= self.delta(m, n, i, j) <= self.D]
        numerator = sum(scores)
        denominator = len(scores)
        # sum(self.B(m, n, k) for k in range(1, self.D + 1))
        assert denominator > 0, (m, n, numerator, self.D)
        return numerator / denominator

    @lru_cache()
    def C(self, n, d):
        return sum(self.c(n, i, j)
                   for i in range(min(d + 1, self.b(n)))
                   for j in range(min(d + 1, self.b(n)))
                   if self.delta(i, j))

    @property
    def D(self):
        d_a = 1
        for d in range(1, max(self.b(m) for m in range(self.n_contigs))):
            if self.E(d) == 0:
                break
            if sum(max(self.b(m) - d, 0) for m in range(self.n_contigs)) < 30:
                break
            d_a = d
        return d_a
        # return min(np.argmax([sum(self.B(u, u, d)>=30 for u in range(self.n_contigs))
        # for d in range(100)]),
        #    np.argmin([sum(self.C(u, v) for u in range(self.n_contigs) for v in range(1, d+1))/sum()

    @lru_cache()
    def score_vector(self):
        return [self.s(m, m + 1) for m in range(self.n_contigs - 1)]


class BinnedBayes(Yahs):
    def __init__(self, count_matrix, contig_start_stops):
        self._count_matrix = count_matrix
        px.imshow(count_matrix)
        self._contig_start_stops = contig_start_stops

    @lru_cache()
    def rate_ratio(self):
        bin_counts = np.sum(self._count_matrix, axis=1)
        rates = bin_counts * bin_counts[:, np.newaxis]
        return self.noise_ratio() / np.mean(rates)

    @lru_cache()
    def rate_non_edge(self, m, n, i, j):
        # return np.sqrt(self.binned_noise(m, i) * self.binned_noise(n, j))
        # return self.noise_ratio()
        count_mul = self.bin_counts(m, i) * self.bin_counts(n, j)
        alpha = 0.9  # self.noise_ratio()
        return alpha * (count_mul * self.rate_ratio()) + (1 - alpha) * self.noise_ratio()

    @lru_cache
    def binned_noise(self, m, i):
        return (self.bin_counts(m, i) - self.contig_matrix(m)[i].sum()) / (
            len(self._count_matrix) - len(self.contig_matrix(m)))

    @lru_cache()
    def bin_counts(self, m, i):
        return self._count_matrix[self._contig_start_stops[m][0] + i, :].sum()

    @property
    def total_count(self):
        return self._count_matrix.sum()

    @lru_cache()
    def noise_ratio(self):
        intra_sum = sum(self.contig_matrix(m).sum() for m in range(self.n_contigs))
        intra_size = sum(self.contig_matrix(m).size for m in range(self.n_contigs))
        # also subtract the intercells
        total_sum = self._count_matrix.sum()
        return (total_sum - intra_sum) / (self._count_matrix.size - intra_size)

    @lru_cache()
    def ratio(self, d):

        ratios = []
        for m in range(self.n_contigs):
            for i in range(self.b(m) - d):
                ratios.append(self.c(m, i, i + d) / self.rate_non_edge(m, m, i, i + d))
        # px(name = 'splitting').histogram(ratios, nbins=100, title=f'distance{d}')
        return np.mean(ratios)

    @lru_cache()
    def signal_rate(self, d):
        '''
        count_ij = Poisson(noise_rate_ij + signal_rate_d(i, j)*donw_sampling_factor_i*down_sampling_factor_j)
        '''
        diffs = [(self.c(m, i, i + d) - self.rate_non_edge(m, m, i, i + d)) for m
                 in range(self.n_contigs)
                 for i in range(self.b(m) - d)]
        ds = [self.down_sampling_factor(m, m, i, i + d) for m
              in range(self.n_contigs)
              for i in range(self.b(m) - d)]
        local_estimates = [d / ds for d, ds in zip(diffs, ds)]
        px.histogram(diffs, nbins=100, title=f'distance{d}')
        px.histogram(local_estimates, nbins=100, title=f'L-distance{d}')
        return np.median(local_estimates)

    @lru_cache()
    def median_bin_count(self):
        return np.median(self._count_matrix.sum(axis=0))

    @lru_cache()
    def log_prob_count_given_non_edge(self, m, n, i, j):
        return self.prob(self.c(m, n, i, j), self.rate_non_edge(m, n, i, j))  # self.noise_ratio())

    @lru_cache()
    def band_probability(self, m, n, i, j):
        pass

    @lru_cache()
    def rate_edge(self, m, n, i, j):
        return self.rate_non_edge(m, n, i, j) + self.signal_rate(self.delta(m, n, i, j)) * self.down_sampling_factor(m,
                                                                                                                     n,
                                                                                                                     i,
                                                                                                                     j)
        # return self.rate_non_edge(m, n, i, j) * self.ratio(self.delta(m, n, i, j))

    def prob(self, count, rate):
        return scipy.stats.poisson.logpmf(count, rate)

    @lru_cache()
    def log_prob_count_given_edge(self, m, n, i, j):
        return self.prob(self.c(m, n, i, j), self.rate_edge(m, n, i, j))

    @lru_cache()
    def triangle_indices(self, m, n):
        return [(i, j) for i in range(max(self.b(m) - self.D - 1, 0), self.b(m))
                for j in range(min(self.b(n), self.D + 1)) if 1 <= self.delta(m, n, i, j) <= self.D]

    def all_intra_indices(self, d):
        return [(m, m) + idx for m in range(self.n_contigs) for idx in self.intra_indices(m, d)]

    def intra_indices(self, m, d):
        return [(i, i + d) for i in range(self.b(m) - d)]

    @lru_cache()
    def s(self, m, n):
        assert m + 1 == n
        indices = self.triangle_indices(m, n)
        likelihood_edge = sum(self.log_prob_count_given_edge(m, n, i, j) for i, j in indices)
        likelihood_non_edge = sum(self.log_prob_count_given_non_edge(m, n, i, j) for i, j in indices)
        log_prob = likelihood_edge - np.logaddexp(likelihood_non_edge, likelihood_edge)
        print(likelihood_edge, likelihood_non_edge, log_prob)
        return log_prob

    def plot(self):
        px.scatter(self._count_matrix.sum(axis=1) / 2, title='bin_counts')
        px.histogram(self._count_matrix.sum(axis=1) / 2, title='bin_counts')
        height = self.D + 1
        edge_matrix = np.zeros((height, (self.D + 1) * self.n_contigs), dtype=float)

        non_edge_matrix = np.zeros_like(edge_matrix)
        count_matrix = np.zeros_like(edge_matrix)
        rate_matrix = np.zeros_like(edge_matrix)
        rate_non_matrix = np.zeros_like(edge_matrix)
        table = defaultdict(list)
        for m in range(self.n_contigs - 1):
            n = m + 1
            assert self._contig_start_stops[m][1] == self._contig_start_stops[n][0], self._contig_start_stops
            indices = [(i, j) for i in range(max(self.b(m) - self.D - 1, 0), self.b(m))
                       for j in range(min(self.D + 1, self.b(n)))
                       if 1 <= self.delta(m, n, i, j) <= self.D]
            for i, j in indices:
                table['m'].append(m)
                table['d'].append(self.delta(m, n, i, j))
                table['rate'].append(self.rate_edge(m, n, i, j))
                table['count'].append(self.c(m, n, i, j))
                x = self.b(m) - i - 1
                y = m * (self.D + 1) + j
                edge_matrix[x, y] = self.log_prob_count_given_edge(m, n, i, j)
                non_edge_matrix[x, y] = self.log_prob_count_given_non_edge(m, n, i, j)
                count_matrix[x, y] = self.c(m, n, i, j)
                rate_matrix[x, y] = self.rate_edge(m, n, i, j)
                rate_non_matrix[x, y] = self.rate_non_edge(m, n, i, j)
        df = pd.DataFrame(table)
        df['ratio'] = df['count'] / df['rate']
        df['bin_size'] = [self.b(m) for m in df['m']]
        df.to_csv('summary.csv', index=False)
        print(df)
        px.scatter(df, x='rate', y='ratio', color='bin_size', facet_col='d')
        f = lambda v: np.log2(v + 1)

        px.imshow(f(count_matrix))
        px.imshow(f(rate_matrix))
        px.imshow(f(rate_non_matrix))
        px.imshow(edge_matrix)
        # px.imshow(indicator_matrix)
        px.imshow(non_edge_matrix)
        px.imshow(edge_matrix - np.logaddexp(edge_matrix, non_edge_matrix))

    @lru_cache()
    def down_sampling_factor(self, m, n, i, j):
        return min(self.bin_counts(m, i) / self.mean_bin_count(), 1) * min(
            self.bin_counts(n, j) / self.median_bin_count(), 1)

    @lru_cache()
    def mean_bin_count(self):
        return np.mean(self._count_matrix.sum(axis=0))

    @lru_cache()
    def sorted_intra_counts(self, d):
        return np.sort([self.c(*idx[1:]) for idx in self.all_intra_indices(d)])

    @lru_cache()
    def qs(self, m, n, i, j):
        sorted_intra_counts = self.sorted_intra_counts(self.delta(m, n, i, j))
        return np.searchsorted(sorted_intra_counts, self.c(m, n, i, j)) / len(sorted_intra_counts)
