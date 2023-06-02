import pickle

import numpy as np
import pandas as pd
import typer
from scipy.stats import poisson, norm

from bnp_assembly.graph_objects import NodeSide
from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
import plotly.express as px
from sklearn.linear_model import PoissonRegressor, LinearRegression

def main(table_name, edge_info_name):
    edges = pickle.load(open(edge_info_name, 'rb'))
    non_edges = edges['inter_chromsome_splits']
    non_edges = [tuple(int(name[6:]) for name in pair) for pair in non_edges]
    boudry_node_sides = [n for edge in non_edges for n in (NodeSide(edge[0], 'r'), NodeSide(edge[1], 'l'))]
    print(boudry_node_sides)
    table = pd.read_csv(table_name)
    boundry_mask = np.array([str(node) in {str(n) for n in boudry_node_sides} for node in table['node_side']])
    print(table.length[boundry_mask])
    table['boundary'] = boundry_mask
    table['z'] = table['node_side_count']-table['length']

    x = table['length']
    X = np.log(x.to_numpy()[:, None])
    y = table['node_side_count']

    poisson_regression_model = PoissonRegressor(alpha=0)
    poisson_regression_model.fit(X, y)
    rate = poisson_regression_model.predict(X)
    print(rate)
    px.scatter(rate, y).show()
    p = poisson.logpmf(y, rate)
    print(p)
    table['p'] = p
    #     px.histogram(table, x='p', color='boundary', barmode='overlay').show()

    linear_regression_model = np.polyfit(x, y, 1)
    residuals = y - np.polyval(linear_regression_model, x)
    z_score = residuals/np.sqrt(np.polyval(linear_regression_model, x))
    table['z_score'] = z_score
    px.histogram(table, x='z_score', color='boundary', barmode='overlay').show()
    # px.scatter(x=x, y=residuals/np.polyval(linear_regression_model, x)).show()
    return table


if __name__ == '__main__':
    # typer.run(main)
    result = main(
        '../../hic-assembly-benchmarking/node_side_counts.csv',
        '../../hic-assembly-benchmarking/data/sacCer3/simulated/big/10/100000/not_assembled/10/hifiasm.hic.p_ctg.fa.edge_info'
    )
