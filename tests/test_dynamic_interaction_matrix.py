from bnp_assembly.location import Location, LocationPair
from bnp_assembly.dynamic_bin_distance_matrix import InteractionMatrixFactory
import numpy as np
import pytest



@pytest.fixture
def contig_dict():
    return {0: 10, 1: 11}


@pytest.fixture
def contig_dict_large():
    return {0: 10, 1: 11, 2: 12, 3: 20}


@pytest.fixture
def location_a():
    return Location.from_entry_tuples([(0, 3), (0, 4)])


@pytest.fixture
def location_b():
    return Location.from_entry_tuples([(1, 5), (1, 6)])


def test_interaction_matrix_factory(contig_dict, location_a, location_b):
    mat = InteractionMatrixFactory(contig_dict, 4).create_from_location_pairs(LocationPair(location_a, location_b))
    data = mat.data
    true = [[0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]

    np.testing.assert_equal(data, true)
    
def test_interaction_matrix_factory(contig_dict):
    factory = InteractionMatrixFactory(contig_dict, 4)
    zero_dict = {0: 0, 1: 0, 2: 0, 3: 0,
                 4: 1, 5: 1, 6: 1,
                 7: 2, 8: 2, 9: 2}
    one_dict = {0: 3, 1: 3, 2: 3, 3: 3,
                4: 4, 5: 4, 6: 4, 7: 4,
                8: 5, 9: 5, 10: 5}

    print(factory._bin_size_dict)
    for offset, bin_id in zero_dict.items():
        print(offset, factory.get_bin(0, offset), bin_id)
        assert factory.get_bin(0, offset) == bin_id

    for offset, bin_id in one_dict.items():
        print(offset, factory.get_bin(0, offset), bin_id)
        assert factory.get_bin(1, offset) == bin_id
        

@pytest.mark.parametrize("bin_size", [2, 4, 5, 6])
def test_splitting_score_with_dynamic_interaction_matrix(contig_dict_large, bin_size):
    contig_dict = contig_dict_large
    from_locations = []
    to_locations = []
    for contig in contig_dict:
        for i in range(contig_dict[contig]):
            for j in range(contig_dict[contig]):
                from_locations.append((contig, i))
                to_locations.append((contig, j))

    from_locations = Location.from_entry_tuples(from_locations)
    to_locations = Location.from_entry_tuples(to_locations)
    location_pairs = LocationPair(from_locations, to_locations)

    bin_size = 4
    factory = InteractionMatrixFactory(contig_dict, bin_size)
    matrix = factory.create_from_location_pairs(location_pairs)

    print(matrix.data)

    for contig_id in list(contig_dict.keys())[1:]:
        score = matrix.get_triangle_score(factory.get_bin(contig_id, 0), max_offset=10)
        assert score == 0