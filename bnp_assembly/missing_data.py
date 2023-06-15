from .location import LocationPair
from typing import Dict, Tuple
import itertools
import numpy as np




def find_regions_with_missing_data(contig_dict: Dict[int, int], read_pairs: LocationPair, bin_size=100) -> Dict[str, Tuple]:
    """
    Finds regions with missing data.
    """
    counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, read_pairs)

    average_bin_count = np.median(np.concatenate([count/bin_size for count in counts.values()]))
    threshold = average_bin_count / 100
    bins_with_missing_data = {contig_id: np.where(counts[contig_id] / bin_sizes[contig_id] < threshold)[0] for contig_id in counts}

    positions_with_missing_data = {contig_id: [] for contig_id in counts}

    for contig, contig_size in bins_with_missing_data.items():
        for bin in bins_with_missing_data[contig]:
            positions_with_missing_data[contig].append((bin * bin_size, (bin+1)*bin_size))

    return positions_with_missing_data


def get_binned_read_counts(bin_size, contig_dict, read_pairs):
    counts = {contig: np.zeros((length + bin_size - 1) // bin_size) for contig, length in contig_dict.items()}
    actual_bin_sizes = {
        contig_id: np.array([min(contig_size, (i+1) * bin_size) - i*bin_size for i in range(len(counts[contig_id]))])
                             for contig_id, contig_size in contig_dict.items()
    }
    assert all(np.all(bin_sizes > 0) for bin_sizes in actual_bin_sizes.values())

    for read in itertools.chain(read_pairs.location_a, read_pairs.location_b):
        counts[int(read.contig_id)][read.offset // bin_size] += 1
    return counts, actual_bin_sizes




