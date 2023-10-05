'''
Run through all reads for intra to calc cumulative distance distribution
Run through all reads for inter to calculate F = dynamic bin size heatmap
Compare edge F's to F's calulated from sampled intra reads.
* Find wich distance matches each bin best and do median or something
'''
import dataclasses

from .distance_distribution import distance_dist
from .plotting import px
from typing import Tuple, Callable, Iterable, Dict, List, Union, Optional
import numpy as np
from dataclasses import dataclass

from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.input_data import FullInputData, NumericInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair, Location
import bionumpy as bnp

import numpy as np

from bnp_assembly.graph_objects import Edge
from bnp_assembly.input_data import FullInputData
from bnp_assembly.location import LocationPair
import logging
logger = logging.getLogger(__name__)


def get_gap_distances(max_gap_distance, n):
    divisor = min(max_gap_distance / (n - 1), 500)
    nth_root = np.power(max_gap_distance // divisor, 1 / (n - 1))
    gaps = np.array([int(divisor * np.power(nth_root, i)) for i in range(n-1)])
    # ignore gaps larger than max distance / 2
    gaps = np.array([g for g in gaps if g < max_gap_distance] + [max_gap_distance])
    assert np.all(gaps[1:] > 0), gaps
    print("Gaps found", gaps)
    return gaps


@dataclass
class DynamicHeatmapConfig:
    scale_func: Callable[[int], int] = lambda x: np.log(x+1).astype(int)
    n_bins: int = 100
    max_distance: int = 100000


log_config = DynamicHeatmapConfig(n_bins=10)


class DynamicHeatmap:
    def __init__(self, array, scale_func=lambda x: np.log(x + 1)):
        self._array = np.asanyarray(array)
        self._scale_func = scale_func

    @property
    def array(self):
        return self._array

    def set_array(self, a):
        self._array = a

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


class HeatmapComparison:
    def __init__(self, heatmap_stack: List[DynamicHeatmap]):
        """
        Heatmap stack is list of heatmaps, biggest gap first
        """
        self._heatmap_stack = np.maximum.accumulate([h.array for h in heatmap_stack], axis=0)
        assert ~np.any(np.isnan(self._heatmap_stack)), self._heatmap_stack
        for i, stack in enumerate(self._heatmap_stack):
            px(name="dynamic_heatmaps").imshow(stack, title=f"Heatmap stack {i}")

    @classmethod
    def from_hetamap_stack(cls, h):
        return cls(h)

    def locate_heatmap(self, heatmap: DynamicHeatmap, plot_name=None):
        idxs = [np.searchsorted(self._heatmap_stack[:, i, j], value)
                for (i, j), value in np.ndenumerate(heatmap.array)]

        if plot_name is not None:
            px(name="dynamic_heatmaps").imshow(np.array(idxs).reshape(heatmap.array.shape), title=plot_name)

        best = np.median(idxs)
        if best >= len(self._heatmap_stack):
            best = len(self._heatmap_stack) - 1
        assert ~np.any(np.isnan(best)), (best, idxs)
        return best

    def get_heatmap_score(self, heatmap: DynamicHeatmap, plot_name=None):
        """
        Returns the score after matching heatmap against stack. Low score is good.
        """
        best_match = self.locate_heatmap(heatmap, plot_name=plot_name)
        score = len(self._row_sums) - best_match
        assert score >= 0
        assert score <= len(self._row_sums)
        return score


class HeatmapComparisonRowColumns(HeatmapComparison):
    """
    Compares heatmaps by looking at row/column sums of values
    """
    def __init__(self, row_sums, col_sums):
        self._row_sums = row_sums
        self._col_sums = col_sums

        for i, (row, col) in enumerate(zip(self._row_sums, self._col_sums)):
            px(name="dynamic_heatmaps").bar(row[:, -1], title=f"Heatmap rows {i}")
            px(name="dynamic_heatmaps").bar(col[-1, :], title=f"Heatmap columns {i}")

    @classmethod
    def from_heatmap_stack(cls, heatmap_stack, add_n_extra=0):
        for i, heatmap in enumerate(heatmap_stack):
            px(name="dynamic_heatmaps").imshow(heatmap.array, title=f"Backgorund heatmap {i}")
            px(name="dynamic_heatmaps").array(heatmap.array, title=f"Backgorund heatmap {i}")

        row_sums = np.array([np.cumsum(heatmap.array, axis=-1) for heatmap in heatmap_stack])
        col_sums = np.array([np.cumsum(heatmap.array, axis=-2) for heatmap in heatmap_stack])

        d_rows = row_sums[0] / add_n_extra  # (row_sums[-1] - row_sums[0]) / (len(row_sums)-1)
        d_cols = col_sums[0] / add_n_extra  # (col_sums[-1] - col_sums[0]) / (len(col_sums)-1)


        cur_row_sums = row_sums[0]
        cur_col_sums = col_sums[0]
        remaining_extra = add_n_extra
        new_cols = []
        new_rows = []
        while (np.any(cur_row_sums[:, -1] >= 1) or np.any(cur_col_sums[-1, :] >= 1)) and remaining_extra>0:
            #cur_row_sums = np.maximum(cur_row_sums-1, 0)
            #cur_col_sums = np.maximum(cur_col_sums-1, 0)
            cur_row_sums = np.maximum(cur_row_sums-d_rows, 0)
            cur_col_sums = np.maximum(cur_col_sums-d_cols, 0)

            new_rows.append(cur_row_sums)
            new_cols.append(cur_col_sums)
            remaining_extra -= 1

        if len(new_rows):
            row_sums = np.concatenate([new_rows[::-1], row_sums])
            col_sums = np.concatenate([new_cols[::-1], col_sums])

        row_sums = np.maximum.accumulate(row_sums, axis=0)
        col_sums = np.maximum.accumulate(col_sums, axis=0)

        return cls(row_sums, col_sums)

    def locate_heatmap(self, heatmap: DynamicHeatmap, plot_name=None):
        row_sums = np.sum(heatmap.array, axis=-1)
        col_sums = np.sum(heatmap.array, axis=-2)

        n_rows, n_cols = heatmap.array.shape

        row_idx = [np.searchsorted(self._row_sums[:, i, n_cols-1], row_sums[i]) for i in range(n_rows)]
        #ds  = [self._row_sums[row_idx+1, i, n_cols-1]-self._row_sums[row_idx, i, n_cols-1] for i in range(n_rows)]
        #row_idx+=[(row_sums[i]-self._row_sums[row_id, i, n_cols-1])/d for i, (row_id, d) in enumerate(zip(row_idx, ds))]
        col_idx = [np.searchsorted(self._col_sums[:, n_rows-1, j], col_sums[j]) for j in range(n_cols)]

        scores = np.concatenate([row_idx, col_idx])
        best = np.mean(scores)
        if plot_name is not None:
            px(name="dynamic_heatmaps").bar(scores, title=plot_name)
            print(plot_name, scores, "Median:", best)

        if best >= len(self._row_sums):
            best = len(self._row_sums) - 1
        return best


class DynamicHeatmaps:
    def __init__(self, size_array: np.ndarray, heatmap_config: DynamicHeatmapConfig = log_config):
        self._size_array = size_array
        self._n_nodes = self._size_array.size
        self._scale_func = heatmap_config.scale_func
        self._n_bins = heatmap_config.n_bins
        self._max_distance = heatmap_config.max_distance
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
                mask = (a_pos < self._max_distance) & (b_pos < self._max_distance)
                a_idx = self._scale_func(a_pos[mask])
                b_idx = self._scale_func(b_pos[mask])
                idx = np.ravel_multi_index(
                    (a.contig_id[mask], b.contig_id[mask],
                     a_idx, b_idx), self._array.shape[2:])
                self._array[a_dir, b_dir] += np.bincount(idx, minlength=self._array[0, 0].size).reshape(self._array.shape[2:])

    def get_heatmap(self, edge: Edge):
        a_dir, b_dir = (0 if node_side.side == 'l' else 1 for node_side in (edge.from_node_side, edge.to_node_side))
        a, b = (node_side.node_id for node_side in (edge.from_node_side, edge.to_node_side))
        a_bins = max(1, self._scale_func(self._size_array[a]))
        b_bins = max(1, self._scale_func(self._size_array[b]))
        assert a_bins >= 1 and b_bins >= 1
        heatmap = DynamicHeatmap(self._array[a_dir, b_dir, a, b, :a_bins,:b_bins], self._scale_func)
        heatmap2 = DynamicHeatmap(self._array[b_dir, a_dir, b, a, :b_bins,:a_bins], self._scale_func)
        return DynamicHeatmap(heatmap.array + heatmap2.array.T, self._scale_func)

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
    dynamic_heatmaps = DynamicHeatmaps(size_array, dynamic_heatmap_config)
    for i, location_pair in enumerate(next(input_data.location_pairs)):
        dynamic_heatmaps.register_location_pairs(location_pair)
    return dynamic_heatmaps


class DynamicHeatmapDistanceFinder(EdgeDistanceFinder):
    def __init__(self, heatmap_config: DynamicHeatmapConfig = log_config, max_gap_distance=None):
        self._heatmap_config = heatmap_config
        self._max_gap_distance = max_gap_distance

    def __call__(self, reads: PairedReadStream, effective_contig_sizes):
        """
        Returns a DirectedDistanceMatrix with "distances" using the dynamic heatmap method
        (comparing heatmaps for contigs against background heatmaps)
        """

        #edge_distances = get_distance_counts_using_dynamic_heatmaps(NumericInputData(effective_contig_sizes, reads))
        input_data = NumericInputData(effective_contig_sizes, reads)
        assert isinstance(input_data.location_pairs, PairedReadStream), type(input_data.location_pairs)
        #                                                                   max_distance=max_distance)
        dynamic_heatmap_creator = PreComputedDynamicHeatmapCreator(input_data.contig_dict, self._heatmap_config, max_gap_distance=self._max_gap_distance)
        sampled_heatmaps = dynamic_heatmap_creator.create(input_data.location_pairs, n_extra_heatmaps=0)
        #for gap, heatmap in sampled_heatmaps.items():
        #    px(name="dynamic_heatmaps").imshow(heatmap.array, title=f"Sampled dynamic heatmap {gap}")

        gap_sizes = list(sampled_heatmaps.keys())
        heatmap_comparison = HeatmapComparisonRowColumns.from_heatmap_stack(list(sampled_heatmaps.values())[::-1], add_n_extra=3)
        heatmaps = get_dynamic_heatmaps_from_reads(self._heatmap_config, input_data)

        distances = {}
        for edge in get_all_possible_edges(len(input_data.contig_dict)):
            heatmap = heatmaps.get_heatmap(edge)
            plot_name = None
            if edge.from_node_side.node_id == edge.to_node_side.node_id - 1:
                plot_name = f"Searcshorted indexes {edge}"
            #distance = gap_sizes[len(gap_sizes) - int(heatmap_comparison.locate_heatmap(heatmap, plot_name=plot_name)) - 1]
            distance = heatmap_comparison.get_heatmap_score(heatmap, plot_name=plot_name)
            distances[edge] = distance

            if edge.from_node_side.node_id == edge.to_node_side.node_id - 1:
                px(name="dynamic_heatmaps").imshow(heatmap.array, title=f"Edge heatmap {edge}")
                print(edge, distance)

        DirectedDistanceMatrix.from_edge_dict(len(input_data.contig_dict), distances).plot(
            name="dynamic_heatmap_scores").show()

        return DirectedDistanceMatrix.from_edge_dict(len(effective_contig_sizes), distances)


class PreComputedDynamicHeatmapCreator:
    """
    Precomputes dynamic heatmaps by using reads fand rom suitable contigs
    * Finds contigs that are large enough to use for estimation
    * Iterates possible gaps between contigs
    * Emulates two smaller contig on chosen contigs and uses these to estimate counts
    """
    def __init__(self, genome: Dict[int, int], heatmap_config: DynamicHeatmapConfig = log_config, n_precomputed_heatmaps=10, max_gap_distance=None):
        self._config = heatmap_config
        self._contig_sizes = genome
        self._size_array = np.zeros(max(self._contig_sizes.keys())+1, dtype=int)
        self._size_array[list(self._contig_sizes.keys())] = list(self._contig_sizes.values())
        if max_gap_distance is None:
            max_gap_distance = self._config.max_distance

        self._gap_distances = get_gap_distances(max_gap_distance, n_precomputed_heatmaps)
        self._chosen_contigs = self._get_suitable_contigs_for_estimation()
        self._chosen_contig_mask = np.zeros(max(self._contig_sizes.keys())+1, dtype=bool)
        self._chosen_contig_mask[self._chosen_contigs] = True
        print("Found %d contigs to estimate heatmaps from" % len(self._chosen_contigs))
        assert len(self._chosen_contigs) > 0, "Did not find any contigs to estimate background dynamic heatmaps. Try setting max distance lower"

    def _get_suitable_contigs_for_estimation(self):
        # find contigs that are at least 2 x max distance
        contigs = [contig for contig, size in self._contig_sizes.items() if size >= 2 * self._config.max_distance + self._gap_distances[-1]]
        if len(contigs) == 0:
            logging.error(self._config.max_distance)
            logging.error(self._gap_distances)
            logging.error(list(self._contig_sizes.values()))
            logging.error("Did not find enough contigs to estimate background dynamic heatmaps. Try setting max distance lower")
            raise Exception("")
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
        return self.get_dynamic_heatmaps(read_pairs, [gap_distance])[0]
        heatmap = DynamicHeatmap.empty(self._config)
        for chunk in read_pairs:
            self.update_heatmap(chunk, gap_distance, heatmap)

        heatmap.set_array(heatmap.array / len(self._chosen_contigs))
        return heatmap

    def get_dynamic_heatmaps(self, read_pairs: Iterable[LocationPair], gap_distances: List[int]) -> List[DynamicHeatmap]:
        heatmaps = [DynamicHeatmap.empty(self._config) for gap in gap_distances]
        for chunk in read_pairs:
            for heatmap, gap_distance in zip(heatmaps, gap_distances):
                self.update_heatmap(chunk, gap_distance, heatmap)
        for heatmap in heatmaps:
            heatmap.set_array(heatmap.array / len(self._chosen_contigs))
        return heatmaps

    def update_heatmap(self, chunk, gap_distance, heatmap):
        pair = chunk
        mask = (self._chosen_contig_mask[pair.location_a.contig_id]) & (self._chosen_contig_mask[pair.location_b.contig_id])
        mask &= pair.location_a.contig_id == pair.location_b.contig_id
        pair = pair.__class__(pair.location_a[mask], pair.location_b[mask])
        min_location, max_location = (np.minimum(pair.location_a.offset, pair.location_b.offset),
                                      np.maximum(pair.location_a.offset, pair.location_b.offset))
        contig_id = pair.location_a.contig_id
        split_positions = self._size_array[contig_id] // 2
        mask = (min_location < split_positions - gap_distance) & (max_location >= split_positions + gap_distance)
        mask &= (min_location>split_positions-gap_distance-self._config.max_distance) & (max_location<split_positions+gap_distance+self._config.max_distance)
        chunk = LocationPair(Location(contig_id, min_location), Location(contig_id, max_location))
        chunk = chunk.subset_with_mask(mask)
        for pair in chunk:
            contig_id_a = int(pair.location_a.contig_id)
            offset_a = int(pair.location_a.offset)
            offset_b = int(pair.location_b.offset)

            PreComputedDynamicHeatmapCreator.add_contig_offset_pair_to_heatmap(
                heatmap, offset_a, offset_b, gap_distance, self._contig_sizes[contig_id_a]
            )

    def create(self, reads: Union[PairedReadStream, Iterable[LocationPair]], n_extra_heatmaps=0) -> DynamicHeatmaps:
        """
        Creates dynamic heatmaps with gaps. If n_extra_heatmaps > 0, also adds n extra heatmaps by "fading" the last one to zero
        """
        print("Using gap sizes %s" % self._gap_distances)
        heatmaps = self.get_dynamic_heatmaps(next(reads), self._gap_distances)
        last_bin, last_heatmap = len(heatmaps)-1, heatmaps[-1]
        # for bin, gap in enumerate(self._gap_distances):
        #     print("Creating heatmap for gap %d" % gap)
        #     # todo: vectorize by adding chunk directly to the correct heatmap
        #     heatmap = self.get_dynamic_heatmap(next(reads), gap)
        #     heatmaps[bin] = heatmap
        #     last_heatmap = heatmap
        #     last_bin = bin
        heatmaps = dict(enumerate(heatmaps))
        if n_extra_heatmaps > 0:
            for i in range(n_extra_heatmaps):
                if np.all(heatmaps[last_bin].array == 0):
                    break
                heatmaps[last_bin+i] = DynamicHeatmap(np.maximum(0, last_heatmap.array - 1))
                #print("Creating extra dynamic heatmap", i)
                #print(heatmaps[last_bin+i].array, np.all(heatmaps[last_bin+i].array == 0))
                last_heatmap = heatmaps[last_bin+i]

        return heatmaps


def get_all_possible_edges(n_contigs):
    all_edges = (Edge.from_numeric_index((i, j)) for i in range(n_contigs*2) for j in range(n_contigs*2))
    return (edge for edge in all_edges if edge.from_node_side.node_id != edge.to_node_side.node_id)


def find_bins_with_even_number_of_reads(cumulative_distance_distribution, n_bins=10, max_distance=1000000) -> DynamicHeatmapConfig:
    """
    Finds n_bins bins that give approx even number of reads in them up to max_distance
    """
    if max_distance >= len(cumulative_distance_distribution):
        max_distance = len(cumulative_distance_distribution)-1

    n_per_bin = cumulative_distance_distribution[max_distance] / n_bins
    cumulative_per_bin = np.cumsum(np.zeros(n_bins) + n_per_bin)
    split_positions = np.searchsorted(cumulative_distance_distribution, cumulative_per_bin[:-1])
    #assert split_positions[-1] == max_distance, f"Last split position {split_positions[-1]} should be {max_distance}"
    bin_borders = np.concatenate([[0], split_positions, [max_distance]])
    return bin_borders


def find_bins_with_even_number_of_reads2(cumulative_distance_distribution, n_bins=10, max_distance=1000000) -> DynamicHeatmapConfig:
    """
    Finds n_bins bins that give approx even number of reads in them up to max_distance
    """
    if max_distance >= len(cumulative_distance_distribution):
        max_distance = len(cumulative_distance_distribution)-1

    bin_offsets = [0]
    total = max_distance*cumulative_distance_distribution[max_distance]
    desired = total/(n_bins**2)
    for i in range(1, max_distance):
        last_offset = bin_offsets[-1]
        bin_size = i - last_offset
        p_distance = cumulative_distance_distribution[2*last_offset+bin_size]-cumulative_distance_distribution[2*last_offset]
        number = bin_size*p_distance
        if number > desired:
            bin_offsets.append(i)
    bin_borders = np.array(bin_offsets)[:n_bins]
    return np.append(bin_borders, max_distance)


def find_bins_with_even_number_of_reads3(cumulative_distance_distribution, n_bins=10, max_distance=1000000) -> DynamicHeatmapConfig:
    """
    Finds n_bins bins that give approx even number of reads in them up to max_distance
    """
    if max_distance > len(cumulative_distance_distribution)//2:
        max_distance = len(cumulative_distance_distribution)//2
    c = cumulative_distance_distribution
    bin_offsets = [0]
    total = np.sum(c[np.arange(max_distance)+max_distance]-c[np.arange(max_distance)])

#    total = cumulative_distance_distribution[max_distance]*max_distance
    # total = max_distance*cumulative_distance_distribution[max_distance]
    desired = total/n_bins
    number = 0
    for i in range(1, max_distance):
        last_offset = bin_offsets[-1]
        bin_size = i - last_offset
        # xs = np.arange(last_offset, last_offset+bin_size)
        # number = np.sum(c[xs+max_distance]-c[xs])
        end = last_offset+bin_size
        number+=c[max_distance+end-1]-c[end-1]
        # number = (cumulative_distance_distribution[last_offset+max_distance]-cumulative_distance_distribution[last_offset])*bin_size
        # p_distance = cumulative_distance_distribution[2*last_offset+bin_size]-cumulative_distance_distribution[2*last_offset]
        # number = bin_size*p_distance
        if number >= desired:
            bin_offsets.append(i)
            number = 0
    bin_borders = np.array(bin_offsets)[:n_bins]
    return np.append(bin_borders, max_distance)

def get_dynamic_heatmap_config_with_even_bins(cumulative_distance_distribution, n_bins=5, max_distance=1000000):
    """
    Creates a dynamic heatmap config with a scale function created by trying to get equal number of
    reads in each bin
    """
    if max_distance >= len(cumulative_distance_distribution):
        max_distance = len(cumulative_distance_distribution)-1
    bin_borders = find_bins_with_even_number_of_reads3(cumulative_distance_distribution, n_bins, max_distance)
    #bin_size = 1000
    #bin_borders = np.arange(n_bins+1) * bin_size
    print("Bin borders: ", bin_borders)

    def scale_func(x):
        if isinstance(x, int) and x >= max_distance:
            # allow ints outside, these should be the last bin
            return n_bins - 1

        return np.searchsorted(bin_borders, x, side='right') - 1

    return DynamicHeatmapConfig(
        scale_func=scale_func,
        n_bins=n_bins,
        max_distance=max_distance
    )


def get_dynamic_heatmap_config_with_uniform_bin_sizes(n_bins=5, bin_size=1000):
    max_distance = n_bins * bin_size

    def scale_func(x):
        if isinstance(x, int) and x >= max_distance:
            # allow ints outside, these should be the last bin
            return n_bins - 1

        return x // bin_size

    return DynamicHeatmapConfig(
        scale_func=scale_func,
        n_bins=n_bins,
        max_distance=max_distance
    )

