import logging
import plotly.express as px
from bnp_assembly.evaluate_scaffold import score_scaffolding, run_simulated_experiment, get_distance_matrix
from bnp_assembly.simulation.hic import SimulationParams
from dataclasses import replace
import numpy as np
logging.basicConfig(level='INFO')


def plot_distance(simulation_params):
    rng = np.random.default_rng(1000)
    d = get_distance_matrix(simulation_params, rng, distance_measure='forbes')
    d.plot().show()
    # px.histogram(d.data.ravel()).show()
    rng = np.random.default_rng(1000)
    true_paths, paths = run_simulated_experiment(simulation_params, rng, distance_measure='forbes')
    print([path.directed_nodes for path in paths])
    print([[d[e] for e in path.edges] for path in paths])

    rng = np.random.default_rng(1000)
    get_distance_matrix(simulation_params, rng, distance_measure='window').plot().show()
    true_paths, paths = run_simulated_experiment(simulation_params, rng, distance_measure='window')
    print([path.directed_nodes for path in paths])
    print([[d[e] for e in path.edges] for path in paths])


plot_distance(SimulationParams(20, 2000, 1000, mean_distance=100))
