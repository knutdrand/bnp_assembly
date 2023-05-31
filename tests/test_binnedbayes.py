import numpy as np
import pytest
from bnp_assembly.plotting import  ResultFolder, register
register(splitting=ResultFolder('./plogging'))
from bnp_assembly.evaluate_scaffold import run_simulated_split_experiment, SplittingParams
from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
from bnp_assembly.simulation.hic import SimulationParams
from bnp_assembly.splitting import YahsSplitter


@pytest.mark.parametrize('n_reads', [100000]) # , 500, 100])
@pytest.mark.parametrize('n_nodes', [10])
def test_ratios(n_reads, n_nodes):

    YahsSplitter.matrix_class = BinnedBayes
    rng = np.random.default_rng(seed=100)
    splitting_params = SplittingParams(50, 0.5)
    simulation_params = SimulationParams(n_nodes, n_reads, n_chromosomes=2, node_length=1000, mean_distance=100)
    true_paths, paths = run_simulated_split_experiment(simulation_params, rng, splitting_params)
    print('predicted', [path.directed_nodes for path in paths])
    print('true', [path.directed_nodes for path in true_paths])
    assert true_paths == paths

