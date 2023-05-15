from bnp_assembly.forbes_score import get_pscore_matrix, get_forbes_matrix, get_pair_counts, get_node_side_counts
from scipy.stats import poisson
from bnp_assembly.graph_objects import Edge, NodeSide
from bnp_assembly.location import Location, LocationPair
import numpy as np
import pytest


@pytest.fixture
def pair_counts():
    return {Edge(NodeSide(0, 'r'), NodeSide(1, 'l')): 2,
            Edge(NodeSide(0, 'l'), NodeSide(1, 'r')): 1,
            Edge(NodeSide(0, 'r'), NodeSide(1, 'r')): 1,
            Edge(NodeSide(0, 'l'), NodeSide(1, 'l')): 0}


@pytest.fixture
def contig_dict():
    return {0: 20, 1: 20}


@pytest.fixture
def locations_pair():
    return LocationPair(
        Location.from_entry_tuples([(0, 15), (0, 16), (0, 5),  (0, 17),  (0, 0),  (1, 9)]),
        Location.from_entry_tuples([(1, 5),  (1, 5),  (1, 15), (1, 17), (0, 10), (1, 19)]))

@pytest.fixture
def node_side_counts(pair_counts):
    return {node_side: sum(value for edge, value in pair_counts.items() 
                           if node_side in (edge.from_node_side, edge.to_node_side))
            for node_side in (NodeSide(i, d) for i in (0, 1) for d in ('l', 'r'))}


@pytest.mark.xfail
def test_pvalue(pair_counts, node_side_counts):
    p_values = get_pscore_matrix(pair_counts, node_side_counts)
    assert p_values[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == poisson.sf(2, 3*2/8)


# @pytest.mark.xfail
def test_forbes(pair_counts, node_side_counts):
    forbes = get_forbes_matrix(pair_counts, node_side_counts, alpha=0)
    assert forbes[Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))] == -np.log(2/(3*2/8))
    assert forbes[Edge(NodeSide(0, 'l'), NodeSide(1, 'r'))] == -np.log(1/(1*2/8))
    assert forbes[Edge(NodeSide(0, 'r'), NodeSide(1, 'r'))] == -np.log(1/(3*2/8))
    assert forbes[Edge(NodeSide(0, 'l'), NodeSide(1, 'l'))] == -np.log(0)


def test_node_side_counts(pair_counts, node_side_counts):
    pair_counts.update({edge.reverse(): count for edge, count in pair_counts.items()})
    assert get_node_side_counts(pair_counts) == node_side_counts


def test_get_pair_counts(locations_pair, pair_counts, contig_dict):
    result = get_pair_counts(contig_dict, locations_pair)
    print(result)
    for key, value in pair_counts.items():
        assert result[key] == value
    assert sum(result.values()) == sum(pair_counts.values())*2
