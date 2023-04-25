import pytest
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices


@pytest.fixture
def contig_list():
    return {0: 100, 1: 50}

@pytest.fixture
def location_pairs():
    location_a = Location.from_entry_tuples([
        (0, 10),
        (1, 40),
        (0, 8)
])
    location_b = Location.from_entry_tuples([
        (1, 45),
        (0, 5),
        (0, 20)])
    return LocationPair(location_a, location_b)


def test_distance_acceptance(contig_list, location_pairs):
    graph = calculate_distance_matrices(contig_list, location_pairs)
