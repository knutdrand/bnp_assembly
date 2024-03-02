import numpy as np
import scipy

from bnp_assembly.plotting import px
from .datatypes import StreamedGenomicLocationPair, GenomicLocationPair
from .location import LocationPair


class CumulativeDist2d:
    def __init__(self, cumulative_dist, background_value=None):
        self._cumulative_dist = cumulative_dist
        self._cumulative_cumulative_dist = np.cumsum(cumulative_dist)
        self._dist_len = len(self._cumulative_dist)
        if background_value is None:
            n = self._dist_len // 10
            self._background_value = (self._cumulative_dist[-1]-self._cumulative_dist[-n-1])/n
        else:
            self._background_value = background_value
        # assert self._background_value>0

    def get_cumcum(self, value):
        if value >= self._dist_len:
            return self._cumulative_cumulative_dist[-1]+self._background_value*(value-self._dist_len)
        return self._cumulative_cumulative_dist[value]

    def get_w_weight(self, distance, w_1, w_2):
        assert w_1>0 and w_2>0
        a = self.get_cumcum(distance + w_1 + w_2)
        b = self.get_cumcum(distance + w_1)
        c = self.get_cumcum(distance + w_2)
        d = self.get_cumcum(distance)
        tmp1 = a - b
        w = tmp1 + d- c

        #assert w != 0, (distance, w_1, w_2, w, a, b, c, d, tmp1)
        return max(w, self._background_value*(w_1*w_2))

    def get_weight(self, start_a, end_a, start_b, end_b):
        t = self.get_cumcum[end_a+end_b]
        t -= self.get_cumcum[end_a+start_b]
        t -= self.get_cumcum[start_a+end_b]
        t += self.get_cumcum[start_a+start_b]
        return t


def distance_dist(location_pairs: object, contig_dict: object) -> np.ndarray:

    if isinstance(location_pairs, LocationPair):
        distances = get_intra_distances(location_pairs)
    else:
        distances = (get_intra_distances(location_pairs) for location_pairs in location_pairs)

    return calculate_distance_distritbution(list(contig_dict.values()), distances)


def get_intra_distances(location_pairs):
    a, b = location_pairs.location_a, location_pairs.location_b
    return np.abs(a.offset - b.offset)[a.contig_id == b.contig_id]


def calculate_distance_distritbution(contig_sizes, distances) -> np.ndarray:
    """
    Returns the cumulative distribution of distances between reads within contigs (intra-reads)
    """
    N = max(contig_sizes)

    if isinstance(distances, (np.ndarray, list)):
        occurances = np.bincount(distances, minlength=N)
    else:
        occurances = sum(np.bincount(d, minlength=N) for d in distances)

    assert np.sum(occurances) > 0, "No occurences of intra reads. No reads within contigs?"

    oppurtunity = np.zeros(N)
    for contig_size in contig_sizes:
        oppurtunity[:contig_size] += 1
    oppurtunity = np.cumsum(oppurtunity[::-1])[::-1]
    ratios = np.cumsum(occurances / oppurtunity)
    ratios /= ratios[-1]
    return ratios


class DistanceDistribution:
    def __init__(self, log_probabilities):
        self._log_probabilities = log_probabilities
        assert np.all(~np.isinf(self._log_probabilities)), np.flatnonzero(np.isinf(self._log_probabilities))

    def cut_at_distance(self):
        return self.__class__(self._log_probabilities[:self.max_distance])

    @property
    def array(self):
        return self._log_probabilities

    def normalize(self):
        return self.__class__(self._log_probabilities - scipy.special.logsumexp(self._log_probabilities))

    @property
    def max_distance(self):
        return len(self._log_probabilities)-1

    @classmethod
    def from_probabilities(cls, probabilities):
        return cls(np.log(probabilities))

    def smooth(self):
        base = self._log_probabilities
        smoothed = scipy.ndimage.gaussian_filter1d(base, 100)
        self._smoothed = smoothed
        return
        max_distance = len(self._log_probabilities)-1
        px(name='joining').line(smoothed, title='smoothed')
        smoothed[-1] = 0
        for i in range(1, len(smoothed) // max_distance + 1):
            s = slice(i * max_distance, (i + 1) * max_distance)
            smoothed[s] = np.mean(smoothed[s])
        smoothed = smoothed + 0.000001 / len(smoothed)
        self._log_probabilities = smoothed / np.sum(smoothed)

    @classmethod
    def from_cumulative_distribution(cls, cumulative_distribution, max_distance=100000):
        base = np.diff(cumulative_distribution)
        smoothed = scipy.ndimage.gaussian_filter1d(base, 10)
        px(name='joining').line(smoothed, title='smoothed')
        smoothed[-1] = 0
        for i in range(1, len(smoothed) // max_distance + 1):
            s = slice(i * max_distance, (i + 1) * max_distance)
            smoothed[s] = np.mean(smoothed[s])
        smoothed = smoothed + 0.000001 / len(smoothed)
        px(name='joining').line(smoothed / np.sum(smoothed), title='smoothed2')
        return cls.from_probabilities(smoothed / np.sum(smoothed))

    @classmethod
    def from_read_pairs(cls, read_pairs, contig_dict):
        return cls.from_cumulative_distribution(distance_dist(read_pairs, contig_dict))

    def log_probability(self, distance):
        distance = np.asanyarray(distance)
        log_probs = self._log_probabilities[np.where(distance < len(self._log_probabilities), distance, -1)]
        return log_probs


    @classmethod
    def load(cls, filename):
        return cls(np.load(filename))

    def save(self, filename):
        np.save(filename, self._log_probabilities)

    def plot(self):
        px(name='joining').line(self._log_probabilities, title='distance distribution')
