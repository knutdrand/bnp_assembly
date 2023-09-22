from typing import Dict

import numpy as np

from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.location import LocationPair
from bnp_assembly.orientation_weighted_counter import OrientationWeightedCounter


def get_forbes_counts(read_pairs, contig_dict, cumulative_distribution, max_distance=100000):
    forbes_obj = OrientationWeightedCounter(contig_dict,
                                            cumulative_length_distribution=cumulative_distribution,
                                            max_distance=max_distance)
    if isinstance(read_pairs, LocationPair):
        read_pairs = [read_pairs]
    for chunk in read_pairs:
        forbes_obj.register_location_pairs(chunk)
    return forbes_obj.counts


class ForbesDistanceFinder(EdgeDistanceFinder):
    def __init__(self, contig_sizes: Dict[int, int], cumulative_distribution: np.ndarray, max_distance: int):
        self._contig_sizes = contig_sizes
        self._cumulative_distribution = cumulative_distribution
        self._max_distance = max_distance

    def __call__(self, reads):
        return get_forbes_counts(reads,
                                 self._contig_sizes,
                                 self._cumulative_distribution,
                                 self._max_distance)
