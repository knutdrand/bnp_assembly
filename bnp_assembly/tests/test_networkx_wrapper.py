# from bnp_assembly.networkx_wrapper import networkx_wrapper
import networkx as nx
from bnp_assembly.graph_objects import NodeSide
import pytest


@pytest.fixture
def node_sides():
    return [[NodeSide(i, d) for d in ('l', 'r')] for i in range(2)]


def test_create_graph(node_sides):
    g = nx.Graph()
    g.add_weighted_edges_from([
        (node_sides[0][i], node_sides[1][j], 1+i+j/2)
        for i in range(2) for j in range(2)])
    assert isinstance(nx.max_weight_matching(g), set)
                                                                
                                                             
