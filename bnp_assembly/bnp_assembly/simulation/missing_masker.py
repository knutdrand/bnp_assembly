from typing import Tuple, List

import numpy as np

from bnp_assembly.simulation.paired_read_positions import PairedReadPositions


def is_in_tuples(postion_a: int, missing_tuples_a: List[Tuple[int, int]]):
    return any(start <= postion_a <= end for start, end in missing_tuples_a)


def mask_missing(read_pairs: PairedReadPositions, missing_dict):
    mask_entries = []
    for read_pair in read_pairs:
        missing_tuples_a = missing_dict[read_pair.contig_1]
        missing_tuples_b = missing_dict[read_pair.contig_2]
        mask_entry = is_in_tuples(read_pair.position_1, missing_tuples_a) or is_in_tuples(read_pair.position_2, missing_tuples_b)
        mask_entries.append(not mask_entry)
    mask = np.array(mask_entries)
    return read_pairs[mask]

