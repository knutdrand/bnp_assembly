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


"""
contig_file_name = sys.argv[1]
bam = sys.argv[2]
genome = bnp.Genome.from_file(contig_file_name)
read_stream = PairedReadStream.from_bam(genome, bam, mapq_threshold=20)
input_data = FullInputData(genome, read_stream)
contig_name_translation, numeric_input_data = get_numeric_input_data(input_data)

global_offset = BinnedNumericGlobalOffset.from_contig_sizes(
    numeric_input_data.contig_dict, 500)
print(global_offset._contig_sizes / global_offset._contig_n_bins)
matrix = SparseInteractionMatrix.from_reads(global_offset, numeric_input_data.location_pairs)
print(average_element_distance(matrix.sparse_matrix))
#to_file(matrix, "testmatrix")

#scipy.sparse.save_npz("testmatrix.npz", matrix.sparse_matrix)

#matrix = SparseInteractionMatrix(scipy.sparse.load_npz("testmatrix.npz"), global_offset)
"""
testpath = pickle.load(open("directed_nodes", "rb"))
matrix = from_file("large_interaction_matrix_trimmed.npz")
distance_pmf = estimate_distance_pmf_from_sparse_matrix2(matrix).array
#px.line(distance_pmf[0:30000]).show()
#sys.exit()

matrix = matrix.get_subset_on_contigs(28, 33)
matrix.plot_submatrix(0, matrix.n_contigs-1)
to_file(matrix, "interaction_matrix_test")
sys.exit()

#testpath = testpath[25:28]
print(testpath)
#background = BackgroundMatrix.from_sparse_interaction_matrix(matrix)
#matspy.spy(background.matrix)
#px.imshow(background.matrix).show()
#sys.exit()

#sub = matrix  #matrix.get_subset_on_contigs(0, 10)
#sub.plot_submatrix(4, 5)
#plt.show()

"""
edge_scores = []
smallest_assumed_chromosome = 1000
for i in range(1, sub.n_contigs-1):
    score = sub.edge_score(i, smallest_assumed_chromosome, background_matrix=background)
    edge_scores.append(score)
"""


#testpath = testpath[:21]
testpath = [DirectedNode(contig, '+') for contig in range(matrix.n_contigs)]
np.save("distance_pmf", distance_pmf)
distance_func = lambda dist: -distance_pmf[dist]
dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(matrix)

#matrix = matrix.get_subset_on_contigs(0, 40)
pathmatrix = matrix.get_matrix_for_path(testpath, as_raw_matrix=False)
#testpath = [DirectedNode(contig, '+') for contig in range(matrix.n_contigs)]
#matrix.plot_submatrix(0, 20)

n_contigs = matrix.n_contigs
path_contig_sizes = np.array([matrix.contig_n_bins[contig.node_id] for contig in testpath])
scorer = LogProbSumOfReadDistancesDynamicScores(testpath.copy(), path_contig_sizes, dists_weights, distance_func=distance_func)

#px.imshow(scorer._score_matrix).show()
#print("Score before", scorer.score())
#scorer.move_contig_to_position(18, 4)
#print("Score after", scorer.score())
#px.imshow(scorer._score_matrix).show()
#new_path = scorer.optimize_positions()
#new_path = scorer.optimize_flippings()
print("Score before start", scorer.score())
print("Score 0:2", scorer.compute_edge_score(2, 3))
#to_file(dists_weights, "dists_weights_test")
print(dists_weights.distances[Edge(NodeSide(0, 'r'), NodeSide(2, 'l'))])
print(dists_weights.weights[Edge(NodeSide(0, 'r'), NodeSide(2, 'l'))])
print(dists_weights.distances[Edge(NodeSide(2, 'r'), NodeSide(0, 'l'))])
print(dists_weights.weights[Edge(NodeSide(2, 'r'), NodeSide(0, 'l'))])
print(scorer._path)
px.imshow(scorer._score_matrix).show()
print("Dist before", scorer.distance_between_contigs_in_path(2, 3))
scorer.find_best_position_for_contig(0)
print(scorer._path)
print("Dist after", scorer.distance_between_contigs_in_path(2, 3))
print("Score 1:2", scorer.compute_edge_score(2, 3))
px.imshow(scorer._score_matrix).show()
#scorer.find_best_position_for_contig(5)
#scorer.find_best_position_for_contig(11)
#scorer.find_best_position_for_contig(18)
#new_path = testpath
new_path = scorer._path
print("Old path", testpath)
print("New path", new_path)

new_matrix = matrix.get_matrix_for_path(new_path, as_raw_matrix=False)
new_matrix.plot_submatrix(0, n_contigs-1)
plt.show()

logging.info(f"Old path: {testpath}")
logging.info(f"New path: {new_path}")

sys.exit()

distance_pmf = estimate_distance_pmf_from_sparse_matrix2(matrix).array
logprobreaddist = LogProbSumOfReadDistances(distance_pmf)

testpath = [DirectedNode(contig, '+') for contig in range(len(matrix._global_offset._contig_n_bins))]
testpath[0] = testpath[0].reverse()
matrix = matrix.get_matrix_for_path(testpath, as_raw_matrix=False)

#matrix = from_file("interaction_matrix_2000.npz")
#pmf = estimate_distance_pmf_from_sparse_matrix2(matrix)
#pmf = pmf.array[-500:]
#px.line(x=np.arange(len(pmf)), y=pmf).show()
matrix.plot_submatrix(0, 10)

#matrix.plot_submatrix(0, 6)
#matrix.set_values_below_threshold_to_zero(matrix.median_weight())
#matrix.plot_submatrix(0, 6)
#plt.show()

#matrix = matrix.get_subset_on_contigs(0, 3)
#matrix.plot_submatrix(0, 3)

#subset.plot_submatrix(0, 7)
#plt.show()
#sys.exit()
#print(type(matrix._data))

#matrix.plot_submatrix(0, 20)
#matrix.flip_contig(3)
#matrix.plot_submatrix(0, 6)

n_contigs = len(matrix._global_offset._contig_sizes)
path = [DirectedNode(contig, '+') for contig in range(n_contigs)]

#testpath = path.copy()
#testpath[3] = testpath[3].reverse()
#print("Testpath score", total_element_distance(matrix.get_matrix_for_path(testpath, as_raw_matrix=True)))

#new = matrix.get_matrix_for_path(path)

#optimizer = TotalDistancePathOptimizer(path.copy(), matrix)
evaluation_function = lambda x: -logprobreaddist(x)

optimizer = PathOptimizer(matrix, evaluation_function)
optimizer.init(path)
new_path = optimizer.run()

print(f"New path {new_path}")
new_matrix = matrix.get_matrix_for_path(new_path, as_raw_matrix=False)
new_matrix.plot_submatrix(0, 10)
plt.show()

sys.exit()
