from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.contig_map import ScaffoldMap
from bnp_assembly.location import Location, LocationPair
import pytest

from bnp_assembly.splitting import ScaffoldSplitter


@pytest.fixture
def contig_dict():
    return {0: 100, 1: 50}


@pytest.fixture
def contig_path():
    return ContigPath.from_directed_nodes([DirectedNode(0, '+'), DirectedNode(1, '-')])


@pytest.fixture
def locations_pair():
    return LocationPair(Location.from_entry_tuples([(0, 5)]), Location.from_entry_tuples([(1, 10)]))


def test_split_acceptance(contig_path, contig_dict, locations_pair):
    splitter = ScaffoldSplitter(contig_dict, 100)
    paths = splitter.split(contig_path, locations_pair)
    assert isinstance(paths, list)
    # split_scaffold(contig_path, contig_dict, bin_size, locations_pair)
    # 
    # global_locations_pair = LocationPair(scaffold_map.translate_locations(locations_pair.location_a),
    #                                      scaffold_map.translate_locations(locations_pair.location_b))
    
