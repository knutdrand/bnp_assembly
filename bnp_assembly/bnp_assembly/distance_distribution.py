import numpy as np
import scipy

from bnp_assembly.plotting import px
from .datatypes import StreamedGenomicLocationPair, GenomicLocationPair
from .location import LocationPair

DISTANCE_CUTOFF = 100000


def distance_dist(location_pairs, contig_dict):

    if isinstance(location_pairs, LocationPair):
        distances = get_intra_distances(location_pairs)
    else:
        distances = (get_intra_distances(location_pairs) for location_pairs in location_pairs)

    return calculate_distance_distritbution(list(contig_dict.values()), distances)


def get_intra_distances(location_pairs):
    a, b = location_pairs.location_a, location_pairs.location_b
    return np.abs(a.offset - b.offset)[a.contig_id == b.contig_id]


def calculate_distance_distritbution(contig_sizes, distances):
    N = max(contig_sizes)

    if isinstance(distances, (np.ndarray, list)):
        occurances = np.bincount(distances, minlength=N)
    else:
        occurances = sum(np.bincount(d, minlength=N) for d in distances)

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

    @classmethod
    def from_probabilities(cls, probabilities):
        return cls(np.log(probabilities))

    @classmethod
    def from_cumulative_distribution(cls, cumulative_distribution):
        base = np.diff(cumulative_distribution)
        smoothed = scipy.ndimage.gaussian_filter1d(base, 10)
        px(name='joining').line(smoothed, title='smoothed')
        smoothed[-1] = 0
        for i in range(1, len(smoothed) // DISTANCE_CUTOFF + 1):
            s = slice(i * DISTANCE_CUTOFF, (i + 1) * DISTANCE_CUTOFF)
            smoothed[s] = np.mean(smoothed[s])
        smoothed = smoothed + 0.000001 / len(smoothed)
        px(name='joining').line(smoothed / np.sum(smoothed), title='smoothed2')
        return cls.from_probabilities(smoothed / np.sum(smoothed))

    @classmethod
    def from_read_pairs(cls, read_pairs, contig_dict):
        return cls.from_cumulative_distribution(distance_dist(read_pairs, contig_dict))

    def log_probability(self, distance):
        distance = np.asanyarray(distance)
        return self._log_probabilities[np.where(distance < len(self._log_probabilities), distance, -1)]

    @classmethod
    def load(cls, filename):
        return cls(np.load(filename))

    def save(self, filename):
        np.save(filename, self._log_probabilities)

    def plot(self):
        px(name='joining').line(self._log_probabilities, title='distance distribution')
