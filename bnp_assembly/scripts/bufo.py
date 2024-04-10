import sys
import logging
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split

logging.basicConfig(level=logging.INFO)
import numpy as np
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, InteractionDistancesAndWeights
from bnp_assembly.sparse_interaction_based_distance import get_inter_background, get_intra_background, \
    get_intra_distance_background
from shared_memory_wrapper import from_file
import plotly.express as px
import matplotlib.pyplot as plt

matrix = from_file("../tests/interaction_matrix_bufo.npz")

new_path, path_matrix = greedy_bayesian_join_and_split(matrix, do_splitting=False)
path_matrix.plot()
plt.show()

print(matrix.sparse_matrix.shape)


#b = get_intra_distance_background(matrix, n_samples=500, max_bins=1000)
b = get_intra_background(matrix)
px.imshow(b.mean(axis=0)).show()
np.save("background.npy", b)

sys.exit()

contig_sizes = matrix._global_offset._contig_n_bins
initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]

dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(matrix)
optimizer = LogProbSumOfReadDistancesDynamicScores(matrix, initial_path)
