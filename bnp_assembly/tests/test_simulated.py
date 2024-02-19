import pytest
from bnp_assembly.simulation.hic import simulate_split_contig_reads, SimulationParams
from bnp_assembly.location import LocationPair
from bnp_assembly.make_scaffold import make_scaffold_numeric
from bnp_assembly.evaluate_scaffold import score_scaffolding, run_simulated_experiment, run_simulated_split_experiment
from bnp_assembly.splitting import YahsSplitter
from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
import logging
logging.basicConfig(level='info')

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


@pytest.mark.skip  # skipped while developing method
@pytest.mark.parametrize('n_reads', [100])  #[1000, 100, 50])# , 500, 100])
@pytest.mark.parametrize('n_nodes', [10])  #, 10, 25, 50])
# joining and splitting methods as tuples:
@pytest.mark.parametrize('method', [
    #('forbes3', 'poisson'),
    #('forbes3', 'matrix'),
    ('dynamic_heatmap', 'matrix')
])
# @pytest.mark.parametrize('size', [4, 8, 10, 20, 30])
def test_simulated(n_reads, n_nodes, method):
    rng = np.random.default_rng(seed=100)
    true_paths, paths = run_simulated_experiment(SimulationParams(n_nodes, n_reads),
                                                 rng, distance_measure=method[0],
                                                 splitting_method=method[1]
                                                 )
    print([path.node_sides for path in paths])
    nodes_visited = [node for path in paths for node in path.nodes]
    assert len(nodes_visited) == n_nodes
    assert len(set(nodes_visited)) == n_nodes
    #assert true_paths == paths


@pytest.mark.skip  # skipped while developing method
@pytest.mark.parametrize('n_reads', [50]) # , 500, 100])
@pytest.mark.parametrize('n_nodes', [4])
@pytest.mark.parametrize('method', ['forbes3'])
# @pytest.mark.parametrize('size', [4, 8, 10, 20, 30])
def test_simulated2(n_reads, n_nodes, method):
    rng = np.random.default_rng(seed=100)
    true_paths, paths = run_simulated_experiment(SimulationParams(n_nodes, n_reads),
                                                 rng, distance_measure=method)
    print([path.node_sides for path in paths])
    nodes_visited = [node for path in paths for node in path.nodes]
    assert len(nodes_visited) == n_nodes
    assert len(set(nodes_visited)) == n_nodes


@pytest.mark.skip  # skipped while developing method
@pytest.mark.parametrize('n_reads', [10000]) # , 500, 100])
@pytest.mark.parametrize('n_nodes', [10, 20, 30])
# @pytest.mark.parametrize('method', ['forbes'])
# @pytest.mark.parametrize('size', [4, 8, 10, 20, 30])
def test_simulated_split(n_reads, n_nodes):
    logging.basicConfig(level='info')
    # YahsSplitter.matrix_class = BinnedBayes
    rng = np.random.default_rng(seed=100)
    true_paths, paths = run_simulated_split_experiment(
        SimulationParams(
            n_nodes,
            n_reads,
            n_chromosomes=2,
        ),
        rng,
    )
    print('predicted', [path.directed_nodes for path in paths])
    print('true',[path.directed_nodes for path in true_paths])
    assert true_paths == paths

