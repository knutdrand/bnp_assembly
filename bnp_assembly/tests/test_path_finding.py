import pytest
from bnp_assembly.hic_distance_matrix import DirectedDistanceMatrix
from bnp_assembly.path_finding import best_path, PathFinder
from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.networkx_wrapper import PathFinder


@pytest.fixture
def distance_matrix():
    m = DirectedDistanceMatrix(3)
    data = m.data
    data += 1
    return m


def test_best_path_acceptance(distance_matrix):
    paths = PathFinder(distance_matrix).run()
    # paths = best_path(distance_matrix)
    assert isinstance(paths, list)
    assert all(isinstance(path, ContigPath) for path in paths)
