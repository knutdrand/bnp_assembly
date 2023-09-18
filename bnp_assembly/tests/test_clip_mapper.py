import pytest
from bionumpy.util.testing import assert_bnpdataclass_equal

from bnp_assembly.clip_mapper import ClipMapper
from bnp_assembly.location import LocationPair, Location


@pytest.fixture
def clip_mapper():
    return ClipMapper({0: (0, 10), 1: (5, 15)})


@pytest.fixture
def location_pair():
    return LocationPair(Location([0, 0, 1, 1], [5, 12, 6, 17]),
                        Location([0, 1, 0, 1], [6, 8, 9, 7]))


def test_map_coordinates(clip_mapper, location_pair):
    result = clip_mapper.map_coordinates(location_pair)
    truth = LocationPair(Location([0, 1], [5, 1]),
                         Location([0, 0], [6, 9]))
    assert_bnpdataclass_equal(result.location_a,
                              truth.location_a)

    assert_bnpdataclass_equal(result.location_b,
                              truth.location_b)

