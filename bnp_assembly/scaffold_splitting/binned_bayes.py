import numpy as np
from functools import lru_cache

import typing as tp

from ..interaction_matrix import InteractionMatrix


class BinnedBayes:
    def __init__(self, interaction_matrix: InteractionMatrix):
        self._interaction_matrix = interaction_matrix


class Yahs:
    def __init__(self, count_matrix: np.ndarray,
                 contig_start_stops: tp.Dict[str, tp.Tuple[tp.Any, tp.Tuple[int, int]]]):
        self._count_matrix = count_matrix
        self._contig_start_stops = contig_start_stops
        n_bins = len(self._count_matrix)
        assert all(start<n_bins and stop <= n_bins for start, stop in self._contig_start_stops.values()), (n_bins, self._contig_start_stops)

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
            assert i<self.b(m) and j<self.b(n), (i, j, self.b(m), self.b(n))
            return self.inter_matrix(m, n)[i, j]

    @lru_cache()
    def w(self, *args):
        return self.c(*args) / self.E(self.delta(*args))

    @property
    def n_contigs(self):
        return len(self._contig_start_stops)

    @lru_cache()
    def E(self, d):
        assert 0 < d <= self.D
        values = [self.c(n, i, j)
                  for n in range(self.n_contigs)
                  for i in range(min(d + 1, self.b(n))) for j in
                  range(min(d + 1, self.b(n))) if self.delta(n, i, j) == d]
        result = np.median(values)
        assert result >= 0
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
        print(weights, len(weights))

        assert denominator > 0, (m, n, d, numerator)
        return numerator / denominator

    @lru_cache()
    def s(self, m, n):
        scores = [self.f(m, n, self.delta(m, n, i, j)) * self.w(m, n, i, j) for i in
             range(max(self.b(m) - self.D - 1, 0), self.b(m)) for j in range(min(self.D + 1, self.b(n))) if
             1 <= self.delta(m, n, i, j) <= self.D]
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
        D =  min(20, max(self.b(m) - 1 for m in range(self.n_contigs)))
        assert D>0, (self._count_matrix.shape, self._contig_start_stops)
        return D
        # return min(np.argmax([sum(self.B(u, u, d)>=30 for u in range(self.n_contigs))
        # for d in range(100)]),
        #    np.argmin([sum(self.C(u, v) for u in range(self.n_contigs) for v in range(1, d+1))/sum()

    @lru_cache()
    def score_vector(self):
        return [self.s(m, m + 1) for m in range(self.n_contigs - 1)]
