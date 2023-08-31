from bnp_assembly.iterative_join import  create_merged_graph
from bnp_assembly.hic_distance_matrix import DirectedDistanceMatrix, Edge, NodeSide
from bnp_assembly.contig_graph import ContigGraph, ContigPath
import pytest


@pytest.fixture
def distance_matrix():
    g = DirectedDistanceMatrix(3)
    g[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] = 1
    g[Edge(NodeSide(1, 'r'), NodeSide(2, 'l'))] = 2
    return g


@pytest.fixture
def paths():
    return [ContigPath.from_node_sides([NodeSide(0, 'l'), NodeSide(0, 'r'),
                                        NodeSide(1, 'l'), NodeSide(1, 'r')]),
            ContigPath.from_node_sides([NodeSide(2, 'l'), NodeSide(2, 'r')])]


def test_create_merged_graph(paths, distance_matrix):
    new_distance_matrix, node_paths = create_merged_graph(paths, distance_matrix)
    assert new_distance_matrix[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == 2
    assert node_paths == dict(enumerate(path.node_sides for path in paths))
