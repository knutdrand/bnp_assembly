import typing as tp

import numpy as np
import scipy.special.cython_special


class OrientationDistribution:
    def __init__(self, contig_length_a: int, contig_length_b: int, length_distribution: 'DistanceDistribution'):
        self._contig_length_a = contig_length_a
        self._contig_length_b = contig_length_b
        self._length_distribution = length_distribution
        self._combinations = [('l', 'l'), ('l', 'r'), ('r', 'l'), ('r', 'r')]

    def distance(self, position_a: int, position_b: int, orientation_a: str, orientation_b: str):
        if orientation_a == 'r':
            position_a = self._contig_length_a - position_a - 1
        if orientation_b == 'r':
            position_b = self._contig_length_b - position_b - 1
        return position_a + position_b + 1

    def orientation_distribution(self, position_a: int, position_b: int):
        combination_probabilities = []
        for combination in self._combinations:
            distance = self.distance(position_a, position_b, *combination)
            log_pmf = self._length_distribution.log_probability(distance)
            combination_probabilities.append(log_pmf)
        total = scipy.special.logsumexp(combination_probabilities)
        probs = [np.exp(p - total) for p in combination_probabilities]
        return dict(zip(self._combinations, probs))
