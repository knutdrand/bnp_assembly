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
    for n in n_reads:
        for _ in range(10):
            paths = run_simulated_experiment(replace(simulation_params, n_reads=n), rng)        
            y.append(score_scaffolding(*paths))
            x.append(n)
            # y.append(np.mean(scores))
    px.scatter(x=np.log10(x), y=y).show()


plot_error_rate(SimulationParams(100, 100, 100))
