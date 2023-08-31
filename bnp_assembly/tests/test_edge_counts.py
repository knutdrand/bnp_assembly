import pytest

from bnp_assembly.edge_counts import EdgeCounts
from bnp_assembly.graph_objects import NodeSide, Edge


@pytest.fixture
def edge_counts():
    return EdgeCounts(2)


def test_edge_counts(edge_counts):
    edge = Edge(NodeSide(0, 'l'), NodeSide(1, 'r'))
    edge_counts[edge] += 1
    assert edge_counts[edge] == 1
    edge_counts[edge] += 2
    assert edge_counts[edge] == 3


def test_edge_count_iter(edge_counts):
    assert set(edge_counts.keys()) == {Edge(NodeSide(0, 'l'), NodeSide(0, 'r')),
                                       Edge(NodeSide(0, 'l'), NodeSide(1, 'r')),
                                       Edge(NodeSide(1, 'l'), NodeSide(0, 'r')),
                                       Edge(NodeSide(1, 'l'), NodeSide(1, 'r')),
                                       Edge(NodeSide(0, 'r'), NodeSide(0, 'l')),
                                       Edge(NodeSide(0, 'r'), NodeSide(1, 'l')),
                                       Edge(NodeSide(1, 'r'), NodeSide(0, 'l')),
                                       Edge(NodeSide(1, 'r'), NodeSide(1, 'l')),
                                       Edge(NodeSide(0, 'l'), NodeSide(0, 'l')),
                                       Edge(NodeSide(0, 'l'), NodeSide(1, 'l')),
                                       Edge(NodeSide(1, 'l'), NodeSide(0, 'l')),
                                       Edge(NodeSide(1, 'l'), NodeSide(1, 'l')),
                                       Edge(NodeSide(0, 'r'), NodeSide(0, 'r')),
                                       Edge(NodeSide(0, 'r'), NodeSide(1, 'r')),
                                       Edge(NodeSide(1, 'r'), NodeSide(0, 'r')),
                                       Edge(NodeSide(1, 'r'), NodeSide(1, 'r'))}
