import pytest
from bnp_assembly.simulation.hic import simulate_split_contig_reads
from bnp_assembly.location import LocationPair
from bnp_assembly.scaffold import scaffold
from bnp_assembly.evaluate_scaffold import score_scaffolding
import numpy as np

np.random.seed(100)


def is_correct_edge(edge):
    if edge.from_node_side.node_id == edge.to_node_side.node_id-1:
        if edge.from_node_side.side=='r' and edge.to_node_side.side=='l':
            return True
    elif edge.from_node_side.node_id==edge.to_node_side.node_id+1:
        if edge.from_node_side.side=='l' and edge.to_node_side.side=='r':
            return True
    return False


@pytest.mark.parametrize('n_reads', [1000, 500, 100, 50])# , 500, 100])
@pytest.mark.parametrize('n_nodes', [4, 8, 10, 20, 25, 50])
# @pytest.mark.parametrize('size', [4, 8, 10, 20, 30])
def test_simulated(n_reads, n_nodes):
    rng =  np.random.default_rng(seed=100)
    split_and_pairs = simulate_split_contig_reads(1000, n_nodes, n_reads, rng=rng)
    n_nodes = len(split_and_pairs.split.starts)
    print(split_and_pairs.split)
    paths = scaffold(split_and_pairs.split.get_contig_dict(),
                     LocationPair(split_and_pairs.location_a,
                                  split_and_pairs.location_b),
                     window_size=30)
    for path in paths:
        print('>>>', path.node_sides)
    nodes_visited = [node for path in paths for node in path.nodes]
    # print(nodes_visited)
    assert len(nodes_visited) == n_nodes
    assert len(set(nodes_visited)) == n_nodes
    true_paths = split_and_pairs.split.get_paths()
    score= score_scaffolding(true_paths, paths)
    # 
    # edges = [e for path in paths for e in path.edges]
    # correct = [is_correct_edge(e) for e in edges]
    # print(edges, correct)
    # 
    # score = sum(correct)
    # assert score//(n_nodes-1) > 0.8
    # edge.from_node_side.node_id for edges in path.edges for path in paths}
    # nodes_visited = {edge.from_node_side.node_id for edges in path.edges for path in paths}
    # 
    # 
    # 
    # score = sum(is_correct_edge(e) for e in path.edges)
    # L = len(path.edges)
    # assert L == 9, path.edges
    # assert score==9, path.edges
