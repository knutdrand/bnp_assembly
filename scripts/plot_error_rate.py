import plotly.express as px
from bnp_assembly.evaluate_scaffold import score_scaffolding, run_simulated_experiment
from bnp_assembly.simulation.hic import SimulationParams
from dataclasses import replace
import numpy as np


def plot_error_rate(simulation_params):
    n_reads = (10**(np.arange(2, 9)/2)).astype(int)
    y = []
    rng = np.random.default_rng()
    x = []
    recalls, precisions = ([], [])
    n_iter = 2
    for n in n_reads:
        for _ in range(n_iter):
            paths = run_simulated_experiment(replace(simulation_params, n_reads=n), rng, distance_measure='forbes')
            precision, recall = score_scaffolding(*paths)
            recalls.append(recall)
            precisions.append(precision)
            x.append(np.log10(n))
    px.scatter(dict(x=x, recall=recalls, precision=precisions), x='recall', y='precision', color='x').show()
    # px.scatter(x=np.log10(x), y=y).show()


plot_error_rate(SimulationParams(100, 100, 100))
