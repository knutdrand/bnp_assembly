from typing import Dict

import numpy as np

from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
#from bnp_assembly.graph_objects import Edge
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair


def get_forbes_counts(read_pairs, contig_dict, cumulative_distribution, max_distance=100000):
    from bnp_assembly.orientation_weighted_counter import OrientationWeightedCounter
    forbes_obj = OrientationWeightedCounter(contig_dict,
                                            cumulative_length_distribution=cumulative_distribution,
                                            max_distance=max_distance)
    if isinstance(read_pairs, LocationPair):
        read_pairs = [read_pairs]
    for chunk in read_pairs:
        forbes_obj.register_location_pairs(chunk)
    return forbes_obj.counts


def create_distance_matrix_forbes_counts(n_nodes, pair_counts: Dict['Edge', float], contig_dict = None, pseudo_count=0.01) -> DirectedDistanceMatrix:
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    for edge, value in pair_counts.items():
        assert (not np.isnan(value)) and (not np.isinf(value)), (edge, value)
        if contig_dict is not None:
            size_factor = np.sqrt(contig_dict[edge.from_node_side.node_id] * contig_dict[edge.to_node_side.node_id])
        else:
            size_factor = 1
        score = -np.log((pseudo_count+value) / size_factor)
        assert (not np.isnan(score)) and (not np.isinf(score)), (edge, value, size_factor)
        distance_matrix[edge] = score
        distance_matrix[edge.reverse()] = score
    return distance_matrix


class ForbesDistanceFinder(EdgeDistanceFinder):
    def __init__(self, contig_sizes: Dict[int, int], cumulative_distribution: np.ndarray, max_distance: int):
        self._contig_sizes = contig_sizes
        self._cumulative_distribution = cumulative_distribution
        self._max_distance = max_distance

    def __call__(self, reads: PairedReadStream, effective_contig_sizes=None):
        if effective_contig_sizes is None:
            effective_contig_sizes = self._contig_sizes

        assert isinstance(reads, PairedReadStream)
        return create_distance_matrix_forbes_counts(len(self._contig_sizes), get_forbes_counts(next(reads),
                                 self._contig_sizes,
                                 self._cumulative_distribution,
                                 self._max_distance),
                                    self._contig_sizes)

