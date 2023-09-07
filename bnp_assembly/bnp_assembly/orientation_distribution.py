import typing as tp

import numpy as np
import scipy.special.cython_special


class OrientationDistribution:

    def __init__(self, contig_length_a: int, contig_length_b: int, length_distribution: 'DistanceDistribution'):
        self._contig_length_a = np.asanyarray(contig_length_a)
        self._contig_length_b = np.asanyarray(contig_length_b)
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
        distances = self.distances(position_a, position_b)
        assert len(distances)==4, distances
        combination_probabilities = self._length_distribution.log_probability(
            distances)
        assert len(combination_probabilities)==4, combination_probabilities
        total = scipy.special.logsumexp(combination_probabilities)
        probs = np.exp(combination_probabilities - total)
        assert len(probs)==4, probs
        assert np.allclose(np.sum(probs), 1), np.sum(probs)
        return dict(zip(self._combinations, probs))

    def distance_matrix(self, position_a: np.ndarray, position_b: np.ndarray)-> np.ndarray:
        '''
        input shapes = (n,)
        output shape = (n, 2, 2)
        '''
        position_a, position_b = np.asanyarray(position_a), np.asanyarray(position_b)
        results = np.empty(position_a.shape+ (2, 2), dtype=int)
        results[..., 0, :] = position_a
        results[..., 1, :] = self._contig_length_a-1-position_a
        results[..., :, 0] += position_b
        results[..., :, 1] += self._contig_length_b-1-position_b
        return results+1
        #factors = np.array([1, -1])
        #offsets_a = np.array([[0], [self._contig_length_a-1]]) # n, 2, 1
        # offsets_b = np.array([0, self._contig_length_b-1]) # n, 1, 2
        # return factors[:, None]*position_a+offsets_a + factors*position_b+offsets_b + 1

    def distribution_matrix(self, position_a, position_b):
        position_a, position_b = np.asanyarray(position_a), np.asanyarray(position_b)
        assert position_a.shape == self._contig_length_a.shape
        assert position_b.shape == self._contig_length_b.shape
        distances = self.distance_matrix(position_a, position_b)
        combination_probabilities = self._length_distribution.log_probability(distances)
        total = scipy.special.logsumexp(combination_probabilities)
        probs = np.exp(combination_probabilities - total)
        assert np.allclose(np.sum(probs), 1), probs
        return probs

