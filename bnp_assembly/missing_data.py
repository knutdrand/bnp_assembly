from .location import LocationPair
from typing import Dict, Tuple, Counter
import itertools
import numpy as np


def find_regions_with_missing_data(contig_dict: Dict[int, int], read_pairs: LocationPair, bin_size=100) -> Dict[str, Tuple]:
    """
    Finds regions with missing data.
    """
    counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, read_pairs)

    average_bin_count = np.median(np.concatenate([count / bin_size for count in counts.values()]))
    threshold = average_bin_count / 100
    bins_with_missing_data = {contig_id: np.where(counts[contig_id] / bin_sizes[contig_id] < threshold)[0] for contig_id
                              in counts}

    positions_with_missing_data = {contig_id: [] for contig_id in counts}

    for contig, contig_size in bins_with_missing_data.items():
        for bin in bins_with_missing_data[contig]:
            positions_with_missing_data[contig].append((bin * bin_size, (bin + 1) * bin_size))

    return positions_with_missing_data


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


def adjust_counts_by_missing_data(existing_counts: Counter,
                                  contig_dict: Dict[str, int],
                                  missing_data: Dict[str, Tuple],
                                  cumulative_length_distribution: np.ndarray,
                                  reads_per_bp: float) -> Counter:
    """
    Adjusts the counts in existing_counts by estimating counts from missing data
    """

    for contig_id, regions in missing_data.items():
        contig_size = contig_dict[contig_id]
        for region in regions:
            start, end = region
            midpoint = (end-start)//2 + start

            p_left = 0.5 * (1 - cumulative_length_distribution[midpoint])
            p_right = 0.5 * (1 - cumulative_length_distribution[contig_size-midpoint+1])

            expected_reads_in_region = reads_per_bp * (end-start)

            for dir, prob in zip(['left', 'right'], [p_left, p_right]):
                expected_reads = prob * expected_reads_in_region


