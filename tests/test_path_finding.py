import pytest
from bnp_assembly.hic_distance_matrix import DirectedDistanceMatrix
from bnp_assembly.path_finding import best_path
from bnp_assembly.contig_graph import ContigPath


@pytest.fixture
def distance_matrix():
    m = DirectedDistanceMatrix(3)
    return m


def test_best_path_acceptance(distance_matrix):
    paths = best_path(distance_matrix)
    print(type(paths))
    assert isinstance(paths, list)
    assert all(isinstance(path, ContigPath) for path in paths)
