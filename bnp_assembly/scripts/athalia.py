import sys

import numpy as np
import plotly.express as px
from bnp_assembly.cli import register_logging
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, InteractionDistancesAndWeights
from shared_memory_wrapper import from_file
from bnp_assembly.distance_matrix import DirectedDistanceMatrix

from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.make_scaffold import make_scaffold_numeric
from bnp_assembly.sparse_interaction_based_distance import get_intra_background

matrix = from_file("interaction_matrix_athalia_10000.npz")
interaction_matrix = matrix
matrix_clipping = from_file("interaction_matrix_athalia_1000.npz")

register_logging("logging")
#background_sums = get_intra_background_sums(interaction_matrix)
#px.histogram(background_sums[:, 235, 2]).show()

from bnp_assembly.iterative_path_joining import IterativePathJoiner

joiner = IterativePathJoiner(interaction_matrix)
joiner.run()


sys.exit()
scaffold = make_scaffold_numeric(interaction_matrix=matrix, interaction_matrix_clipping=matrix_clipping)

distance_matrix = DirectedDistanceMatrix.from_matrix(np.load("logging/main/bayesian_edge_probs.npy"))
distance_matrix.plot(name="original").show()

distance_matrix.set_worst_edges_to_zero(set_to=np.max(distance_matrix.data[~np.isinf(distance_matrix.data)]))
distance_matrix.plot(name="worst_edges_zeroed").show()
distance_matrix.plot(dirs="rrll", name="rrll").show()
px.imshow(distance_matrix.data).show()

