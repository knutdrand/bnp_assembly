from typing import Iterable

import numpy as np


def estimate_max_distance2(contig_sizes: Iterable[int]):
    """
    Finds a distance so contigs >  this distance cover at least 10% of total genome
    """
    sorted = np.sort(list(contig_sizes))[::-1]
    cumsum = np.cumsum(sorted)
    total_size = cumsum[-1]
    cutoff = np.searchsorted(cumsum, total_size // 10, side="right")
    return sorted[cutoff] // 8
