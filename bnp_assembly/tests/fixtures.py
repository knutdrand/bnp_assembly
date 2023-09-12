import pytest
from bnp_assembly.location import Location, LocationPair


@pytest.fixture
def contig_dict():
    return {0: 100, 1: 200, 2: 300}


@pytest.fixture
def read_pairs():
    a = Location.from_entry_tuples([(0, 10), (1, 20), (2, 30)])
    b = Location.from_entry_tuples([(0, 20), (1, 32), (1, 41)])
    return LocationPair(a, b)
