import numpy as np


def distance_dist(location_pairs, contig_dict):
    return calculate_distance_distritbution(list(contig_dict.values()),
                                            [np.abs(a.offset - b.offset)
                                             for a, b in zip(location_pairs.location_a, location_pairs.location_b)
                                             if a.contig_id == b.contig_id])


def calculate_distance_distritbution(contig_sizes, distances):
    N = max(contig_sizes)
    occurances = np.bincount(distances, minlength=N)

    oppurtunity = np.zeros(N)
    for contig_size in contig_sizes:
        oppurtunity[:contig_size] += 1
    oppurtunity = np.cumsum(oppurtunity[::-1])[::-1]
    ratios = np.cumsum(occurances / oppurtunity)
    ratios /= ratios[-1]
    return ratios
