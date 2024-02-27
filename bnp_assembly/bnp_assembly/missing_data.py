import logging

from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix

logging.basicConfig(level=logging.INFO)
from npstructures import RaggedArray, RaggedShape

from .graph_objects import NodeSide, Edge
from .io import PairedReadStream
from .location import LocationPair
from typing import Dict, Tuple, List, Iterable
from collections import Counter
import itertools
import numpy as np
import logging

from .util import add_dict_counts
from .plotting import px
from .change_point import find_change_point

logger = logging.getLogger(__name__)


def find_regions_with_missing_data(contig_dict: Dict[int, int], read_pairs: LocationPair, bin_size=100) -> Dict[
    str, Tuple]:
    """
    Finds regions with missing data.
    """
    counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, read_pairs)

    return find_regions_with_missing_data_from_bincounts(bin_size, bin_sizes, counts)


def find_regions_with_missing_data_from_bincounts(bin_size, bin_sizes, counts):
    average_bin_count = np.median(np.concatenate([count / bin_size for count in counts.values()]))
    threshold = average_bin_count / 20
    bins_with_missing_data = {contig_id: np.where(counts[contig_id] / bin_sizes[contig_id] < threshold)[0] for contig_id
                              in counts}
    positions_with_missing_data = {contig_id: [] for contig_id in counts}
    px(name='joining').histogram(np.concatenate([counts[contig_id] / bin_sizes[contig_id] for contig_id in counts]),
                                 title='missing')
    for contig, bins in bins_with_missing_data.items():
        for bin in bins:
            actual_bin_size = bin_sizes[contig][bin]
            positions_with_missing_data[contig].append((bin * bin_size, (bin * bin_size) + actual_bin_size))
    logger.info('Found %d regions with missing data',
                sum(len(positions) for positions in positions_with_missing_data.values()))
    return positions_with_missing_data, average_bin_count


def find_start_and_end_split_site_for_contig(contig_size, contig_missing_regions):
    prev = 0
    for start, end in contig_missing_regions:
        if start > prev:
            break
        prev = end

    start_split = prev
    # split at prev
    # missing region at end
    prev = contig_size
    for start, end in contig_missing_regions[::-1]:
        if end != prev:
            break
        prev = start

    end_split = prev
    start_split = min(end_split, start_split)  # happens if whole contig is missing

    return start_split, end_split


def find_missing_regions_at_start_and_end_of_contigs(contig_dict: Dict[str, int],
                                                     missing_regions: Dict[str, List[Tuple]]) -> Dict[str, Tuple]:
    """
    Returns a dict of contig ids to new start and end positions (where missing regions at end and start are removed)
    """
    split_sites = {}
    for contig in contig_dict:
        contig_size = contig_dict[contig]
        if contig not in missing_regions:
            split_sites[contig] = (0, contig_size)
            continue
        # missing region at start
        contig_missing_regions = missing_regions[contig]
        split_sites[contig] = find_start_and_end_split_site_for_contig(contig_size, contig_missing_regions)

    return split_sites


def get_binned_read_counts(bin_size, contig_dict, read_pairs):
    row_lens = [(length + bin_size - 1) // bin_size for length in contig_dict.values()]
    shape = RaggedShape(row_lens)
    assert max(contig_dict.keys()) == len(contig_dict)-1, contig_dict.keys()
    total_bins = sum(row_lens)
    # counts = RaggedArray(np.zeros(total_bins), row_lens)
    # counts = {contig: np.zeros((length + bin_size - 1) // bin_size) for contig, length in contig_dict.items()}
    actual_bin_sizes = {
        contig_id: np.array(
            [min(contig_size, (i + 1) * bin_size) - i * bin_size for i in range(row_lens[contig_id])])
        for contig_id, contig_size in contig_dict.items()
    }
    assert all(np.all(bin_sizes > 0) for bin_sizes in actual_bin_sizes.values())

    counts = sum(
        np.bincount(shape.ravel_multi_index((locations.contig_id, locations.offset // bin_size)), minlength=total_bins)
        for locations in (read_pairs.location_a, read_pairs.location_b))
    counts = RaggedArray(counts, row_lens)
    counts = {contig_id: counts[contig_id] for contig_id in range(len(contig_dict))}
    # for read in itertools.chain(read_pairs.location_a, read_pairs.location_b):
    #     counts[int(read.contig_id)][read.offset // bin_size] += 1
    return counts, actual_bin_sizes


def find_missing_data_and_adjust(existing_counts: Counter, contig_dict: Dict[str, int], read_pairs: LocationPair,
                                 cumulative_length_distribution,
                                 bin_size,
                                 max_distance):
    regions, reads_per_bp = find_regions_with_missing_data(contig_dict, read_pairs, bin_size)
    return adjust_counts_by_missing_data(existing_counts, contig_dict, regions, cumulative_length_distribution,
                                         reads_per_bp, max_distance=max_distance)


def adjust_counts_by_missing_data(existing_counts: Counter,
                                  contig_dict: Dict[str, int],
                                  missing_data: Dict[str, Tuple],
                                  cumulative_length_distribution: np.ndarray,
                                  reads_per_bp: float,
                                  max_distance: int) -> Counter:
    """
    Adjusts the counts in existing_counts by estimating counts from missing data
    """
    distance_cutoff = max_distance

    adjusted_counts = Counter()
    missing_counts = Counter()
    for contig_id, regions in missing_data.items():
        contig_size = contig_dict[contig_id]
        other_node_sides = [NodeSide(c, direction) for c in contig_dict for direction in ['l', 'r']]
        for region in regions:
            start, end = region
            if start > distance_cutoff and end < contig_size - distance_cutoff:
                continue

            midpoint = (end - start) // 2 + start

            p_left = 0.5 * (1 - cumulative_length_distribution[min(midpoint, len(cumulative_length_distribution) - 1)])
            p_right = 0.5 * (1 - cumulative_length_distribution[max(0, contig_size - midpoint + 1)])

            expected_reads_in_region = reads_per_bp * (end - start)

            for dir, prob in zip(['l', 'r'], [p_left, p_right]):
                expected_reads = prob * expected_reads_in_region
                edges = [Edge(NodeSide(contig_id, dir), other_node_side) for other_node_side in other_node_sides]
                total_counts = sum([existing_counts[edge] for edge in edges])
                edge_proportions = {edge: existing_counts[edge] / total_counts if total_counts > 0 else 0 for edge in
                                    edges}
                for edge in edges:
                    missing_counts[edge] = edge_proportions[edge] * expected_reads
    for edge in existing_counts:
        adjusted_counts[edge] = existing_counts[edge] + missing_counts[edge]

    return adjusted_counts


def find_end_clip(bins, window_size=None, mean_coverage=None):
    return find_start_clip(bins[::-1], window_size, mean_coverage)


def find_start_clip(bins, window_size=None, mean_coverage=None):
    return find_change_point(bins[0:len(bins)])


def find_clips(bins, mean_coverage=None, window_size=None):
    start, end = (find_start_clip(bins),
                  find_end_clip(bins))
    if start + end >= len(bins):
        return (0, 0)  # Whole node disappears, just use the whole node
    return (start, end)


def find_contig_clips(bin_size: int, contig_dict: Dict[str, int], read_pairs: PairedReadStream, window_size=10) \
        -> Dict[int, Tuple[int, int]]:
    assert isinstance(read_pairs, PairedReadStream)
    bins, bin_sizes = get_missing_region_counts(contig_dict, next(read_pairs), bin_size)

    # normalize last bin (which might be smaller)
    for key, array in bins.items():
        array[-1] *= bin_size / bin_sizes[key][-1]

    for node, b in bins.items():
        px(name="missing_data").array(b, title=f"bin counts contig {node}")
        px(name="missing_data").line(b, title=f"bin counts contig {node}")

    mean_coverage = sum(np.sum(counts) for counts in bins.values()) / sum(contig_dict.values()) * bin_size
    # logger.info(f"Mean coverage: {mean_coverage}, bin_size: {bin_sizes}")
    clip_ids = {contig_id: find_clips(counts, mean_coverage / 2, window_size) for contig_id, counts in bins.items()}
    logger.info(f"Bin size when finding clips: {bin_size}")
    logger.info(f"Found clips: {clip_ids}")
    clips = {contig_id: (start_id * bin_size, contig_dict[contig_id] - end_id * bin_size) for
             contig_id, (start_id, end_id) in clip_ids.items()}
    return clips


def find_contig_clips_from_interaction_matrix(contig_dict: Dict[str, int],
                                              interaction_matrix: SparseInteractionMatrix, window_size=10) \
        -> Dict[int, Tuple[float, float]]:

    #bins, bin_sizes = get_missing_region_counts_from_interaction_matrix(contig_dict, interaction_matrix)
    bins = {}
    bin_sizes = {}
    for contig in contig_dict:
        logging.info(f"Processing contig {contig}")
        submatrix = interaction_matrix.contig_submatrix(contig).toarray()
        bins[contig] = interaction_matrix.get_contig_coverage_counts(contig).astype(int)
        bin_sizes[contig] = interaction_matrix.contig_bin_size(contig)
        #px(name="missing_data").imshow(np.log(submatrix+1), title=f"contig matrix {contig}")

    for node, b in bins.items():
        px(name="missing_data").array(b, title=f"bin counts contig {node}")
        px(name="missing_data").line(b, title=f"bin counts contig {node}")

    #mean_coverage = interaction_matrix.mean_coverage_per_bin()  #sum(np.sum(counts) for counts in bins.values()) / sum(contig_dict.values()) * bin_size
    #assert isinstance(mean_coverage, float), mean_coverage
    #logging.info(f"Mean coverage: {mean_coverage}, bin_size: {bin_sizes}")
    # logger.info(f"Mean coverage: {mean_coverage}, bin_size: {bin_sizes}")
    clip_ids = {contig_id: find_clips(counts) for contig_id, counts in bins.items()}
    logger.info(f"Found clips: {clip_ids}")
    clips = {contig_id: (start_id * bin_sizes[contig_id], contig_dict[contig_id] - end_id * bin_sizes[contig_id]) for
             contig_id, (start_id, end_id) in clip_ids.items()}
    return clips


def get_missing_region_counts(contig_dict, read_pairs, bin_size):
    all_counts = Counter()
    if isinstance(read_pairs, LocationPair):
        read_pairs = [read_pairs]
    for chunk in read_pairs:
        local_counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, chunk)
        all_counts = add_dict_counts(all_counts, local_counts)
    return all_counts, bin_sizes
