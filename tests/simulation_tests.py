import pytest
from bnp_assembly.simulation.hic import simulate_split_contig_reads
from bnp_assembly.location import LocationPair
from bnp_assembly.scaffold import scaffold
import numpy as np

np.random.seed(100)


def is_correct_edge(edge):
    if edge.from_node_side.node_id==edge.to_node_side.node_id-1:
        if edge.from_node_side.side=='r' and edge.to_node_side.side=='l':
            return True
    elif edge.from_node_side.node_id==edge.to_node_side.node_id+1:
        if edge.from_node_side.side=='l' and edge.to_node_side.side=='r':
            return True
    return False


@pytest.mark.parametrize('n_reads', [1000, 500, 100])
def test_simulated(n_reads):
    rng =  np.random.default_rng(seed=100)
    split_and_pairs = simulate_split_contig_reads(1000, 10, n_reads, rng=rng)
    print(split_and_pairs.split)

    path = scaffold(split_and_pairs.split.get_contig_dict(),
                    LocationPair(split_and_pairs.location_a,
                                 split_and_pairs.location_b),
                    window_size=20)
    score = sum(is_correct_edge(e) for e in path.edges)
    L = len(path.edges)
    assert L == 9, path.edges
    assert score==9, path.edges
