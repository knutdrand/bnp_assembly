from bnp_assembly.forbes_score import get_pvalue_matrix, get_forbes_matrix
from scipy.stats import poisson
from bnp_assembly.graph_objects import Edge, NodeSide
import pytest


@pytest.fixture
def pair_counts():
    return {Edge(NodeSide(0, 'r'), NodeSide(1, 'l')): 2,
            Edge(NodeSide(0, 'l'), NodeSide(1, 'r')): 1,
            Edge(NodeSide(0, 'r'), NodeSide(1, 'r')): 1,
            Edge(NodeSide(0, 'l'), NodeSide(1, 'l')): 0}


@pytest.fixture
def node_side_counts(pair_counts):
    return {node_side: sum(value for edge, value in pair_counts.items() 
                           if node_side in (edge.from_node_side, edge.to_node_side))
            for node_side in (NodeSide(i, d) for i in (0, 1) for d in ('l', 'r'))}


def test_pvalue(pair_counts, node_side_counts):
    p_values = get_pvalue_matrix(pair_counts, node_side_counts)
    assert p_values[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == poisson.sf(2, 3*2/4)


@pytest.mark.xfail
def test_forbes(pair_counts, node_side_counts):
    p_values = get_forbes_matrix(pair_counts, node_side_counts, alpha=0)
    assert p_values[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == 2/(3*2/4)
