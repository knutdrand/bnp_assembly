import typing as tp

import numpy as np
import scipy.special.cython_special


class OrientationDistribution:

    def __init__(self, contig_length_a: int, contig_length_b: int, length_distribution: 'DistanceDistribution'):
        self._contig_length_a = contig_length_a
        self._contig_length_b = contig_length_b
        self._length_distribution = length_distribution
        self._combinations = [('l', 'l'), ('l', 'r'), ('r', 'l'), ('r', 'r')]

    def distance(self, position_a: int, position_b: int, orientation_a: str, orientation_b: str) -> int:
        """
        Return the distance between two positions on the two contigs, according to the orientations
        Parameters
        ----------
        position_a
        position_b
        orientation_a
        orientation_b

        Returns
        -------

        """
        if orientation_a == 'r':
            position_a = self._contig_length_a - position_a - 1
        if orientation_b == 'r':
            position_b = self._contig_length_b - position_b - 1
        return position_a + position_b + 1

    def distances(self, position_a: int, position_b: int) -> tp.List[int]:
        """
        Return a list of distances for all possible orientations
        Parameters
        ----------
        position_a
        position_b

        Returns
        -------

        """
        return [self.distance(position_a, position_b, *combination) for combination in self._combinations]

    def orientation_distribution(self, position_a: int, position_b: int)-> tp.Dict[tp.Tuple[str, str], float]:
        """
        Return a dictionary of the probabilities of each orientation combination.
        I.e. {('l', 'r'): 0.5, ('r', 'r'): 0.5, ('l', 'l'): 0, ('r', 'l'): 0}
        Parameters
        ----------
        position_a
        position_b

        Returns
        -------

        """

        combination_probabilities = self._length_distribution.log_probability(self.distances(position_a, position_b))
        total = scipy.special.logsumexp(combination_probabilities)
        probs = np.exp(combination_probabilities - total)
        return dict(zip(self._combinations, probs))

    def distance_matrix(self, position_a, position_b):
        a = np.array([[position_a], [self._contig_length_a - position_a - 1]])
        b = [position_b, self._contig_length_b - position_b - 1]
        return a + b

    def distribution_matrix(self, position_a, position_b):
        distances = self.distance_matrix(position_a, position_b)
        combination_probabilities = self._length_distribution.log_probability(distances)
        total = scipy.special.logsumexp(combination_probabilities)
        return np.exp(combination_probabilities - total)
