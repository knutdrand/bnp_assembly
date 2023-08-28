from .distance_distribution import DISTANCE_CUTOFF
from .graph_objects import NodeSide, Edge
from .location import LocationPair
from typing import Dict, Tuple
from collections import Counter
import itertools
import numpy as np
import logging

from .plotting import px

logger = logging.getLogger(__name__)


def find_regions_with_missing_data(contig_dict: Dict[int, int], read_pairs: LocationPair, bin_size=100) -> Dict[
    str, Tuple]:
    """
    Finds regions with missing data.
    """
    counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, read_pairs)

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
            positions_with_missing_data[contig].append((bin * bin_size, (bin * bin_size)+actual_bin_size))
    logger.info('Found %d regions with missing data',
                sum(len(positions) for positions in positions_with_missing_data.values()))
    return positions_with_missing_data, average_bin_count


def get_binned_read_counts(bin_size, contig_dict, read_pairs):
    counts = {contig: np.zeros((length + bin_size - 1) // bin_size) for contig, length in contig_dict.items()}
    actual_bin_sizes = {
        contig_id: np.array(
            [min(contig_size, (i + 1) * bin_size) - i * bin_size for i in range(len(counts[contig_id]))])
        for contig_id, contig_size in contig_dict.items()
    }
    assert all(np.all(bin_sizes > 0) for bin_sizes in actual_bin_sizes.values())

    for read in itertools.chain(read_pairs.location_a, read_pairs.location_b):
        counts[int(read.contig_id)][read.offset // bin_size] += 1
    return counts, actual_bin_sizes


def find_missing_data_and_adjust(existing_counts: Counter, contig_dict: Dict[str, int], read_pairs: LocationPair,
                                 cumulative_length_distribution,
                                 bin_size):
    regions, reads_per_bp = find_regions_with_missing_data(contig_dict, read_pairs, bin_size)
    return adjust_counts_by_missing_data(existing_counts, contig_dict, regions, cumulative_length_distribution,
                                         reads_per_bp)


def adjust_counts_by_missing_data(existing_counts: Counter,
                                  contig_dict: Dict[str, int],
                                  missing_data: Dict[str, Tuple],
                                  cumulative_length_distribution: np.ndarray,
                                  reads_per_bp: float) -> Counter:
    """
    Adjusts the counts in existing_counts by estimating counts from missing data
    """
    distance_cutoff = DISTANCE_CUTOFF

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
        print(edge)
        adjusted_counts[edge] = existing_counts[edge] + missing_counts[edge]

    return adjusted_counts
