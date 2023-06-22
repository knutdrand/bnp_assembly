import numpy as np

from bnp_assembly.distance_distribution import calculate_distance_distritbution, distance_dist
from numpy.testing import assert_array_equal
from .fixtures import contig_dict, read_pairs

def test_distance_distribution():
    '''possibilities = (0, 1), (0, 1), (0, 2), (1, 2)'''
    '''= (1, 1, 2, 1) '''
    sizes = [2, 3]
    distances = [1, 2]
    F = calculate_distance_distritbution(sizes, distances)
    assert_array_equal(F, [0, (1/3)/(4/3), 1])


def test_distance_dist(contig_dict, read_pairs):
    dist = distance_dist(read_pairs, contig_dict)
    d = np.insert(np.diff(dist), 0, 0)
    assert d[10]>0
    assert d[12]>0
    assert d[11]==0
