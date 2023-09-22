'''
Run through all reads for intra to calc cumulative distance distribution
Run through all reads for inter to calculate F = dynamic bin size heatmap
Compare edge F's to F's calulated from sampled intra reads.
* Find wich distance matches each bin best and do median or something
'''
from typing import Tuple, Callable, Iterable, Dict, List, Union
import numpy as np
from dataclasses import dataclass

from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.input_data import FullInputData, NumericInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair
import bionumpy as bnp

import numpy as np

from bnp_assembly.graph_objects import Edge
from bnp_assembly.input_data import FullInputData
from bnp_assembly.location import LocationPair


@dataclass
class DynamicHeatmapConfig:
    scale_func: Callable[[int], int] = lambda x: np.log(x+1).astype(int)
    inverse_scale_func: Callable[[int], int] = lambda x: np.exp(x) - 1
    n_bins: int = 100

    @property
    def max_distance(self):
        return self.inverse_scale_func(self.n_bins)-1


log_config = DynamicHeatmapConfig(n_bins=10)


class DynamicHeatmap:
    def __init__(self, array, scale_func=lambda x: np.log(x + 1)):
        self._array = np.asanyarray(array)
        self._scale_func = scale_func

    @property
    def array(self):
        return self._array

    @classmethod
    def empty(cls, config: DynamicHeatmapConfig):
        return cls(np.zeros((config.n_bins, config.n_bins)), config.scale_func)

    def add(self, a, b):
        self._array[self._scale_func(a), self._scale_func(b)] += 1

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


class HeatmapComparison:
    def __init__(self, heatmap_stack: List[DynamicHeatmap]):
        """
        Heatmap stack is list of heatmaps, biggest gap first
        """
        self._heatmap_stack = np.maximum.accumulate([h.array for h in heatmap_stack], axis=0)

    def locate_heatmap(self, heatmap: DynamicHeatmap):
        idxs = [np.searchsorted(self._heatmap_stack[:, i, j], value)
                for (i, j), value in np.ndenumerate(heatmap.array)]
        return np.median(idxs)


class DynamicHeatmaps:
    def __init__(self, size_array: np.ndarray, n_bins, scale_func=lambda x: np.log(x + 1).astype(int)):
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
        reverse_a_pos = self._size_array[a.contig_id] - a.offset-1
        reverse_b_pos = self._size_array[b.contig_id] - b.offset-1
        assert np.all(b.offset < self._size_array[b.contig_id])
        assert np.all(a.offset < self._size_array[a.contig_id])

        for a_dir, a_pos in enumerate([a.offset, reverse_a_pos]):
            for b_dir, b_pos in enumerate([b.offset, reverse_b_pos]):
                a_idx = self._scale_func(a_pos)
                b_idx = self._scale_func(b_pos)
                mask = (a_idx < self._n_bins) & (b_idx < self._n_bins)
                idx = np.ravel_multi_index(
                    (a.contig_id[mask], b.contig_id[mask],
                     a_idx[mask], b_idx[mask]), self._array.shape[2:])
                self._array[a_dir, b_dir] += np.bincount(idx, minlength=self._array[0, 0].size).reshape(self._array.shape[2:])

    def get_heatmap(self, edge: Edge):
        a_dir, b_dir = (0 if node_side.side == 'l' else 1 for node_side in (edge.from_node_side, edge.to_node_side))
        a, b = (node_side.node_id for node_side in (edge.from_node_side, edge.to_node_side))
        a_bins = self._scale_func(self._size_array[a])
        b_bins = self._scale_func(self._size_array[b])
        return DynamicHeatmap(self._array[a_dir, b_dir, a, b, :a_bins,:b_bins], self._scale_func)

    @property
    def array(self):
        return self._array


def mean_heatmap(heatmaps_array):
    shape = tuple(max(a.shape[i] for a in heatmaps_array) for i in range(2))
    T = np.zeros(shape)
    counts = np.zeros(shape)
    for heatmap in heatmaps_array:
        T[:heatmap.shape[0], :heatmap.shape[1]] += heatmap
        counts[:heatmap.shape[0], :heatmap.shape[1]] += 1
    return T / np.maximum(counts, 1)


def get_dynamic_heatmaps_from_reads(dynamic_heatmap_config, input_data: NumericInputData):
    contig_sizes = input_data.contig_dict
    size_array = np.array(list(contig_sizes.values()))
    dynamic_heatmaps = DynamicHeatmaps(size_array, n_bins=dynamic_heatmap_config.n_bins,
                                       scale_func=dynamic_heatmap_config.scale_func)
    for i, location_pair in enumerate(next(input_data.location_pairs)):
        dynamic_heatmaps.register_location_pairs(location_pair)
        if i > 1000:
            break
    return dynamic_heatmaps


def get_distance_counts_using_dynamic_heatmaps(input_data: NumericInputData) -> Dict[Edge, float]:
    """
    Finds a "distance" for each edge. Returns a by scoring edges by comparing DynamicHeatmaps to presampled dynamic heatmaps
    """
    assert isinstance(input_data.location_pairs, PairedReadStream), type(input_data.location_pairs)
    dynamic_heatmap_config = log_config
    dynamic_heatmap_creator = PreComputedDynamicHeatmapCreator(input_data.contig_dict, dynamic_heatmap_config)
    sampled_heatmaps = dynamic_heatmap_creator.create(input_data.location_pairs, n_precomputed=10)
    gap_sizes = list(sampled_heatmaps.keys())
    heatmap_comparison = HeatmapComparison(list(sampled_heatmaps.values())[::-1])
    heatmaps = get_dynamic_heatmaps_from_reads(dynamic_heatmap_config, input_data)

    distances = {}
    for edge in get_all_possible_edges(len(input_data.contig_dict)):
        heatmap = heatmaps.get_heatmap(edge)
        distance = gap_sizes[len(gap_sizes) - int(heatmap_comparison.locate_heatmap(heatmap)) - 1]
        distances[edge] = distance

    return distances


class DynamicHeatmapDistanceFinder(EdgeDistanceFinder):
    def __init__(self, contig_sizes: Dict[int, int], heatmap_config: DynamicHeatmapConfig = log_config):
        self._heatmap_config = heatmap_config
        self._contig_sizes = contig_sizes

    def __call__(self, reads: PairedReadStream):
        return get_distance_counts_using_dynamic_heatmaps(NumericInputData(self._contig_sizes, reads))


class PreComputedDynamicHeatmapCreator:
    """
    Precomputes dynamic heatmaps by using reads from suitable contigs
    * Finds contigs that are large enough to use for estimation
    * Iterates possible gaps between contigs
    * Emulates two smaller contig on chosen contigs and uses these to estimate counts
    """
    def __init__(self, genome: Dict[int, int], heatmap_config: DynamicHeatmapConfig = log_config):
        self._config = heatmap_config
        self._contig_sizes = genome
        self._chosen_contigs = self._get_suitable_contigs_for_estimation()

    def _get_suitable_contigs_for_estimation(self):
        # find contigs that are at least 2 x max distance
        contigs = [contig for contig, size in self._contig_sizes.items() if size >= 2 * self._config.max_distance]
        return contigs

    @staticmethod
    def add_contig_offset_pair_to_heatmap(heatmap: DynamicHeatmap, offset_a, offset_b, gap: int, contig_size: int):
        """
        offset_a and offset_b are local offsets on contigs, the first should be lowert than the other
        """
        split_position = contig_size // 2
        subcontig_a_end = split_position - gap
        subcontig_b_start = split_position + gap

        assert offset_a < split_position
        assert offset_b >= split_position
        heatmap_offset_a = subcontig_a_end - offset_a - 1
        heatmap_offset_b = offset_b - subcontig_b_start
        assert heatmap_offset_a >= 0
        assert heatmap_offset_b >= 0
        heatmap.add(heatmap_offset_a, heatmap_offset_b)

    def get_dynamic_heatmap(self, read_pairs: Iterable[LocationPair], gap_distance: int) -> DynamicHeatmap:
        heatmap = DynamicHeatmap.empty(self._config)
        for chunk in read_pairs:
            for pair in chunk:
                contig_id_a = int(pair.location_a.contig_id)
                contig_id_b = int(pair.location_b.contig_id)
                if contig_id_a == contig_id_b and pair.location_a.contig_id in self._chosen_contigs:

                    offset_a = int(pair.location_a.offset)
                    offset_b = int(pair.location_b.offset)

                    if offset_b < offset_a:  # swap, lowest offset first
                        offset_a = int(pair.location_b.offset)
                        offset_b = int(pair.location_a.offset)

                    split_position = self._contig_sizes[contig_id_a] // 2
                    if offset_a >= split_position - gap_distance or offset_b < split_position + gap_distance:
                        # read pair not on seperate subcontigs
                        continue

                    if offset_a < split_position - gap_distance - self._config.max_distance:
                        # too far away for heatmap
                        continue

                    if offset_b > split_position + gap_distance + self._config.max_distance:
                        continue

                    PreComputedDynamicHeatmapCreator.add_contig_offset_pair_to_heatmap(
                        heatmap, offset_a, offset_b, gap_distance, self._contig_sizes[contig_id_a]
                    )

        return heatmap

    def create(self, reads: Union[PairedReadStream, Iterable[LocationPair]], n_precomputed) -> DynamicHeatmaps:
        gap_distances = np.array([2 ** (d + 1) for d in range(n_precomputed)])
        heatmaps = {}
        for bin, gap in enumerate(gap_distances):
            heatmap = self.get_dynamic_heatmap(next(reads), gap)
            heatmaps[bin] = heatmap

        return heatmaps


def get_all_possible_edges(n_contigs):
    all_edges = (Edge.from_numeric_index((i, j)) for i in range(n_contigs) for j in range(n_contigs))
    return (edge for edge in all_edges if edge.from_node_side.node_id != edge.to_node_side.node_id)

