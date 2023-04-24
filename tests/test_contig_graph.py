import pytest
import numpy as np
from bnp_assembly import ContigGraph
F = 0
R = 1
'''$-AL-AR CR-CL BL-BR #'''
correct_path = {('A', F), ('C', R), ('B', F)}

nodes = ['A', 'B', 'C']
rl_edges = {'A': {'A': np.inf, 'B': 10, 'C': 5},
            'B': {'A': 10, 'B': np.inf, 'C': 3},
            'C': {'A': 8, 'B': 9, 'C': np.inf}}

rr_edges = {'A': {'A': np.inf, 'B': 2, 'C': 0.2},
            'B': {'A': 2, 'B': np.inf, 'C': 2},
            'C': {'A': 0.2, 'B': 2, 'C': np.inf}}

ll_edges = {'A': {'A': np.inf, 'B': 2, 'C': 3},
            'B': {'A': 2, 'B': np.inf, 'C': 0.1},
            'C': {'A': 4, 'B': 0.1, 'C': np.inf}}


@pytest.fixture
def contig_graph():
    return ContigGraph.from_distance_dicts(rl_edges, rr_edges, ll_edges)


def test_contig_graph():
    graph = ContigGraph.from_distance_dicts(rl_edges, rr_edges, ll_edges)
    assert graph.nodes == ['A', 'B', 'C']
