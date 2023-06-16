from bnp_assembly.expected_edge_counts import CumulativeDistribution, ExpectedEdgeCounts

import numpy as np
import pytest

from bnp_assembly.graph_objects import Edge, NodeSide


@pytest.fixture
def cumulative_distribution():
    ps = np.arange(10) / np.sum(np.arange(10))
    return CumulativeDistribution(ps, p_noise=0.5, genome_size=20)


def test_cdf(cumulative_distribution):
    assert cumulative_distribution.cdf(19) == 1
    assert cumulative_distribution.cdf(9) == 0.75


cumulative_distribution = np.ones(10)
cumulative_distribution[0] = 0

@pytest.fixture()
def expected_edge_counts():
    return ExpectedEdgeCounts(contig_dict={0: 2, 1: 2, 2: 2}, cumulative_distribution=cumulative_distribution)


def test_get_expected_edge_count(expected_edge_counts):
    expected = expected_edge_counts.get_expected_edge_count(Edge(NodeSide(1, 'r'), NodeSide(2, 'l')))
    np.testing.assert_almost_equal(expected, 1/6)

