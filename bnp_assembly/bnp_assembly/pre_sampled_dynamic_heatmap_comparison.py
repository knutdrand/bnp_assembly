'''
Run through all reads for intra to calc cumulative distance distribution
Run through all reads for inter to calculate F = dynamic bin size heatmap
Compare edge F's to F's calulated from sampled intra reads.
* Find wich distance matches each bin best and do median or something
'''
from typing import Tuple, Callable
import numpy as np
from dataclasses import dataclass
from bnp_assembly.input_data import FullInputData
from bnp_assembly.location import LocationPair


@dataclass
class DynamicHeatmapConfig:
    scale_func: Callable[[int], int] = lambda x: np.log(x+1)
    inverse_scale_func: Callable[[int], int] = lambda x: np.exp(x) - 1
    n_bins: int = 100

    @property
    def max_distance(self):
        return self.inverse_scale_func(self.n_bins)-1

log_config = DynamicHeatmapConfig(n_bins=10)


class DynamicHeatmap:
    def __init__(self, array, scale_func):
        self._array = array
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


class DynamicHeatmaps:
    """Represents multiple DynamicHeatmaps in a shared np.array"""
    pass


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



class PreComputedDynamicHeatmapCreator:
    """
    Precomputes dynamic heatmaps by using reads from suitable contigs
    * Finds contigs that are large enough to use for estimation
    * Iterates possible gaps between contigs
    * Emulates two smaller contig on chosen contigs and uses these to estimate counts
    """
    def __init__(self, input_data: FullInputData, heatmap_config: DynamicHeatmapConfig = log_config, n_precomputed: int = 10):
        self._input_data = input_data
        self._gap_distances = np.array([2 ** (d + 1) for d in range(n_precomputed)])
        self._config = heatmap_config
        self._contig_sizes = input_data.genome.get_genome_context().chrom_sizes
        self._chosen_contigs = self._get_suitable_contigs_for_estimation()

    def _get_suitable_contigs_for_estimation(self):
        # find contigs that are at least 2 x max distance
        contigs = [contig for contig, size in self._contig_sizes.items() if size >= 2 * self._config.max_distance]
        return contigs

    @staticmethod
    def add_contig_offset_pair_to_heatmap(heatmap: DynamicHeatmap, offset_a, offset_b, gap: int, contig_size: int):
        split_position = contig_size // 2
        assert offset_a < split_position
        assert offset_b >= split_position
        heatmap_offset_a = (split_position - gap) - offset_a - 1
        heatmap_offset_b = offset_b - (split_position + gap)
        assert heatmap_offset_a >= 0
        assert heatmap_offset_b >= 0
        heatmap.add(heatmap_offset_a, heatmap_offset_b)

    def create_for_gap_distance(self, gap_distance: int) -> DynamicHeatmap:
        heatmap = DynamicHeatmap.empty(self._config)
        read_pairs = next(self._input_data.paired_read_stream)
        for pair in read_pairs:
            if (pair.location_a.contig_id == pair.location_b.contig_id
                    and pair.location_a.contig_id in self._chosen_contigs):
                self.add_read_pair_to_heatmap(heatmap, pair, gap_distance)

    def create(self) -> DynamicHeatmaps:
        for bin, gap in enumerate(self._gap_distances):
            heatmap = self.create_for_gap_distance(gap)


