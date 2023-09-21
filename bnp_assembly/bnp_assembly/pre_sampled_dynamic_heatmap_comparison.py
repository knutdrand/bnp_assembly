'''
Run through all reads for intra to calc cumulative distance distribution
Run through all reads for inter to calculate F = dynamic bin size heatmap
Compare edge F's to F's calulated from sampled intra reads.
* Find wich distance matches each bin best and do median or something
'''
from typing import Tuple

import numpy as np

from bnp_assembly.input_data import FullInputData
from bnp_assembly.location import LocationPair


class DynamicHeatmap:
    def __init__(self, array, scale_func):
        self._array = array
        self._scale_func = scale_func

    @property
    def array(self):
        return self._array

    @classmethod
    def create_from_positions(cls, positions_tuple: Tuple[list, list], scale_func=lambda x: np.log(x + 1),
                              n_bins: int = 100):
        scaled_positions = tuple(scale_func(p) for p in positions_tuple)
        mask_a, mask_b = (scaled_position < n_bins for scaled_position in scaled_positions)
        mask = mask_a & mask_b
        a, b = (scaled_position[mask] for scaled_position in scaled_positions)
        indices = a * n_bins + b
        array = np.bincount(indices, minlength=n_bins ** 2).reshape(n_bins, n_bins)
        return cls(array, scale_func)


class DynamicHeatmaps:
    def __init__(self, size_array: np.ndarray, n_bins, scale_func=lambda x: np.log(x + 1)):
        self._size_array = size_array
        self._n_nodes = self._size_array.size
        self._scale_func = scale_func
        self._n_bins = n_bins
        self._array = np.zeros(
            (2, 2, self._n_nodes, self._n_nodes, self._n_bins, self._n_bins))

    def _get_flat_index(self, tuple_index):
        return sum(i * factor for i, factor in zip(tuple_index, self._array.shape))

    def register_location_pairs(self, location_pairs: LocationPair):
        a = location_pairs.location_a
        b = location_pairs.location_b
        reverse_a_pos = self._size_array[a.contig_id] - a.offset
        reverse_b_pos = self._size_array[b.contig_id] - b.offset
        for a_dir, a_pos in enumerate([a.offset, reverse_a_pos]):
            for b_dir, b_pos in enumerate([b.offset, reverse_b_pos]):
                a_idx = self._scale_func(a_pos)
                b_idx = self._scale_func(b_pos)
                mask = (a_idx < self._n_bins) & (b_idx < self._n_bins)
                idx = np.ravel_multi_index(
                    (a.contig_id[mask], b.contig_id[mask],
                     a_idx[mask], b_idx[mask]), self._array.shape[2:])
                self._array[a_dir, b_dir] += np.bincount(idx, minlength=self._array[0, 0].size).reshape(self._array.shape[2:])


def get_heatmaps_for_edges(input_data, max_distance, n_bins, n_precomputed):
    heatmaps_for_edges = {}
    for edge in input_data.edges:
        edge_reads = input_data.paired_read_stream.get_reads_for_edge(edge)
        intra_reads = [read for read in edge_reads if read.is_intra]
        inter_reads = [read for read in edge_reads if not read.is_intra]
        intra_distance_distribution = get_samplable_distance_distribution(intra_reads)
        inter_distance_distribution = get_samplable_distance_distribution(inter_reads)
        intra_heatmap = get_dynamic_heatmap(intra_distance_distribution, max_distance, n_bins)
        inter_heatmap = get_dynamic_heatmap(inter_distance_distribution, max_distance, n_bins)
        heatmaps_for_edges[edge] = (intra_heatmap, inter_heatmap)
    return heatmaps_for_edges


def method(input_data: FullInputData, max_distance, n_bins, n_precomputed):
    samplable_distance_distribution = get_samplable_distance_distribution(next(input_data.paired_read_stream))
    pre_sampled_heatmaps = {d: get_dynamic_heatmap(sampled_distance_distribution, 2 ** (d + 1), n_bins) for d in
                            range(n_precomputed)}
    pre_sampled_heatmaps = np.array([pre_sampled_heatmaps[d] for d in range(n_precomputed)])
    heatmaps_for_edges = get_heatmaps_for_edges(input_data, max_distance, n_bins, n_precomputed)
