import numpy as np


class PairDistribution:
    def __init__(self, contig_length, p: float):
        self._contig_length = contig_length
        self._p = p

    def sample(self, rng, n_samples=1):
        distance = np.minimum(rng.geometric(self._p, size=n_samples), self._contig_length - 2)
        first = rng.integers(0, self._contig_length - distance)
        second = first + distance
        assert np.all(second < self._contig_length), (second, self._contig_length)
        assert np.all(first < self._contig_length), (first, self._contig_length)
        direction = rng.choice([True, False], size=n_samples)
        return np.where(direction, first, first + distance), np.where(direction, first + distance, first)
