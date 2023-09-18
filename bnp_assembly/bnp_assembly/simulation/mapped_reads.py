import numpy as np

from bnp_assembly.simulation.pair_distribution import SingleContigPairDistribution


class MappedReadPairs:
    def __init__(self, pair_distribution, noise_distribution, p_signal):
        self._pair_distribution = pair_distribution
        self._noise_distribution = pair_distribution
        self._p_signal = p_signal

    def sample(self, n):
        n_signal = np.random.binomial(n, self._p_signal)
        n_noise = n - n_signal
        signal_pairs = self._pair_distribution.sample(n_signal)
        noise_pairs = self._noise_distribution.sample(n_noise)
        return np.concatenate([signal_pairs, noise_pairs])
