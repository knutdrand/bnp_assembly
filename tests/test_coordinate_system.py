import pytest

from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.coordinate_system import CoordinateSystem
from bnp_assembly.location import LocationPair, Location


@pytest.fixture
def contig_dict():
    return {0: 100, 1: 50, 2: 20}


@pytest.fixture
def contig_path():
    return ContigPath.from_directed_nodes([DirectedNode(0, '+'),
                                           DirectedNode(1, '-'),
                                           DirectedNode(2, '+')])


@pytest.fixture
def coordinate_system(contig_dict, contig_path):
    return CoordinateSystem(contig_dict, contig_path.edges)


locations_pair = LocationPair(Location.from_entry_tuples([(0, 5), (1, 10), (1, 10)]), Location.from_entry_tuples([(1, 10), (2, 5), (0, 5)]))
truth = [(94, 39), (10, 5), (94, 39)]

@pytest.mark.xfail
def test_node_side_coordinates(coordinate_system):
    assert coordinate_system.location_coordinates(Location.single_entry(0, 5)) == 94
    assert coordinate_system.location_coordinates(Location.single_entry(1, 10)) == 10

def test_coordinate_system(coordinate_system):
    print(coordinate_system._edges)
    for location_pair, t in zip(locations_pair, truth):
        assert coordinate_system.location_pair_coordinates(location_pair)[1] == t
