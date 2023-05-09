from bnp_assembly.distance_distribution import calculate_distance_distritbution
from numpy.testing import assert_array_equal


def test_distance_distribution():
    '''possibilities = (0, 1), (0, 1), (0, 2), (1, 2)'''
    '''= (1, 1, 2, 1) '''
    sizes = [2, 3]
    distances = [1, 2]
    F = calculate_distance_distritbution(sizes, distances)
    assert_array_equal(F, [0, (1/3)/(4/3), 1])
    
