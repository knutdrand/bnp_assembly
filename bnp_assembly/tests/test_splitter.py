from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.contig_map import ScaffoldMap
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.location import Location, LocationPair
import numpy as np
import pytest

from bnp_assembly.splitting import ScaffoldSplitter
from bnp_assembly.interaction_matrix import get_triangle_score


@pytest.fixture
def contig_dict():
    return {0: 100, 1: 50}


@pytest.fixture
def contig_path():
    return ContigPath.from_directed_nodes([DirectedNode(0, '+'), DirectedNode(1, '-')])


@pytest.fixture
def locations_pair():
    return LocationPair(Location.from_entry_tuples([(0, 5)]), Location.from_entry_tuples([(1, 10)]))

@pytest.fixture
def locations_pair2():
    return LocationPair(
        Location.from_entry_tuples([(0, 5), (1, 10), (1, 10)]),
        Location.from_entry_tuples([(1, 10), (0, 5), (1, 10)]))



def test_split_acceptance(contig_path, contig_dict, locations_pair):
    splitter = ScaffoldSplitter(contig_dict, 100)
    paths = splitter.split(contig_path, locations_pair)
    assert isinstance(paths, list)


@pytest.fixture
def matrix():
    return np.array([[1, 2, 3],
                     [2, 5, 6],
                     [3, 6, 9]])


def test_splitterinterface(contig_dict, locations_pair2, contig_path):
    splitter = SplitterInterface(contig_dict, locations_pair2, contig_path, max_distance=100, bin_size=10)
    splitter.split()
    assert sum(splitter._node_histograms.values()).sum() == 2

def test_splittermatrix(matrix):
    assert get_triangle_score(matrix, bin_n=1, max_offset=1) == 8
    assert get_triangle_score(matrix, bin_n=1, max_offset=2) == 11

    # split_scaffold(contig_path, contig_dict, bin_size, locations_pair)
    #
    # global_locations_pair = LocationPair(scaffold_map.translate_locations(locations_pair.location_a),
    #                                      scaffold_map.translate_locations(locations_pair.location_b))

