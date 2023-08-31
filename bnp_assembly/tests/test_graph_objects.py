from bnp_assembly.graph_objects import NodeSide, Edge
import pytest

@pytest.fixture
def node_side():
    return NodeSide(1, 'r')

@pytest.fixture
def edge():
    return Edge(NodeSide(1, 'r'), NodeSide(2, 'l'))

def test_from_string_edge(edge):
    assert Edge.from_string(repr(edge)) == edge



