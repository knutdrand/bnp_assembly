from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.interface import SplitterInterface, score_matrix
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


def test_score_matrix(expected, bad, hiding_bad):
    bad_score = score_matrix(bad, expected)
    good_score = score_matrix(expected, expected)
    assert bad_score < good_score


def test_score_matrix_hiding(expected, hiding_bad):
    good_score = score_matrix(expected, expected)
    hiding_bad_score = score_matrix(hiding_bad, expected)
    assert hiding_bad_score < good_score


@pytest.fixture
def hiding_bad():
    hiding_bad = np.array([[3, 3, 3],
                           [1, 1, 0],
                           [1, 0, 0]])
    return hiding_bad

@pytest.fixture
def bad():
    bad = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1.]])
    return bad

@pytest.fixture
def expected():
    expected = np.array([[3, 2, 1],
                         [2, 1, 0.1],
                         [1, 0.1, 0.1]])
    return expected


def test_score_incomplete(expected, bad, hiding_bad):
    bad = bad[:2]
    hiding_bad = hiding_bad[:2]
    good = expected.copy()[:2]
    bad_score = score_matrix(bad, expected)
    good_score = score_matrix(good, expected)
    hiding_bad_score = score_matrix(hiding_bad, expected)
    assert bad_score < good_score
    assert hiding_bad_score < good_score



