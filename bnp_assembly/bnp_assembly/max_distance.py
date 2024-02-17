from typing import Iterable

import numpy as np


def estimate_max_distance2(contig_sizes: Iterable[int]):
    """
    Finds a distance so contigs >  this distance cover at least 10% of total genome
    """
    percent_covered = 0.05
    sorted = np.sort(list(contig_sizes))[::-1]
    cumsum = np.cumsum(sorted)
    total_size = cumsum[-1]
    cutoff = np.searchsorted(cumsum, int(total_size)*percent_covered, side="right")
    return sorted[cutoff] // 8
