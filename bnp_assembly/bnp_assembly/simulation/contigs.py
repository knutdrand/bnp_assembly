import numpy as np


class ContigLengths:
    def __init__(self, expected_length):
        self._expected_length = expected_length

    def sample(self, n=1):
        return np.random.poisson(self._expected_length, size=n)


class ContigDict:
    def __init__(self, n_contigs, contig_lengths):
        self._n_contigs = n_contigs
        self._contig_lengths = contig_lengths

    def sample(self):
        sizes = self._contig_lengths.sample(self._n_contigs)
        return {f'contig_{i}': size for i, size in enumerate(sizes)}

