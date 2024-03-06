import logging
import pickle
import time

import matspy
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt
from shared_memory_wrapper import to_file, from_file

from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import TotalDistancePathOptimizer, PathOptimizer, \
    InteractionDistancesAndWeights, LogProbSumOfReadDistancesDynamicScores
from bnp_assembly.graph_objects import NodeSide, Edge

logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
from bnp_assembly.input_data import FullInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.make_scaffold import get_numeric_input_data
from bnp_assembly.sparse_interaction_matrix import NaiveSparseInteractionMatrix, BinnedNumericGlobalOffset, \
    SparseInteractionMatrix, average_element_distance, total_element_distance, estimate_distance_pmf_from_sparse_matrix, \
    estimate_distance_pmf_from_sparse_matrix2, LogProbSumOfReadDistances, BackgroundMatrix
import sys
import scipy
import plotly.express as px
import numpy as np


testpath = pickle.load(open("directed_nodes", "rb"))
matrix = from_file("large_interaction_matrix_trimmed.npz")
distance_pmf = estimate_distance_pmf_from_sparse_matrix2(matrix).array
px.line(distance_pmf[0:30000]).show()
sys.exit()

#matrix = matrix.get_subset_on_contigs(28, 33)
matrix.plot_submatrix(0, matrix.n_contigs-1)

#testpath = [DirectedNode(contig, '+') for contig in range(matrix.n_contigs)]
distance_func = lambda dist: -distance_pmf[dist]
dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(matrix)

pathmatrix = matrix.get_matrix_for_path(testpath, as_raw_matrix=False)

n_contigs = matrix.n_contigs
path_contig_sizes = np.array([matrix.contig_n_bins[contig.node_id] for contig in testpath])
scorer = LogProbSumOfReadDistancesDynamicScores(testpath.copy(), path_contig_sizes, dists_weights, distance_func=distance_func)

print("Score before start", scorer.score())
#px.imshow(scorer._score_matrix).show()
scorer.optimize_positions()
scorer.optimize_flippings()
print(scorer._path)
px.imshow(scorer._score_matrix).show()
new_path = scorer._path
to_file(new_path, "new_path")
logging.info(f"Old path: {testpath}")
logging.info(f"New path: {new_path}")

new_matrix = matrix.get_matrix_for_path(new_path, as_raw_matrix=False)

pos_old_path = 4
pos_new_path = 26
assert testpath[pos_old_path].node_id == 4
assert new_path[pos_new_path].node_id == 4

new_matrix.plot_submatrix(0, n_contigs-1)
plt.show()


sys.exit()
