import sys

import logging
import plotly
from bnp_assembly.bayesian_splitting import bayesian_split
from bnp_assembly.cli import register_logging
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.missing_data import find_contig_clips_from_interaction_matrix
from bnp_assembly.sparse_interaction_matrix import BackgroundInterMatrices
import plotly.express as px
logging.basicConfig(level=logging.INFO)
from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, \
    InteractionDistancesAndWeights, split_using_inter_distribution
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split
from shared_memory_wrapper import from_file, to_file
import matplotlib.pyplot as plt

#matrix = from_file("../tests/interaction_matrix_bufo.npz")
#inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(matrix, max_bins=5000)
#matrix = from_file("heatmap-contig2377.png.matrix.npz")
#matrix = from_file("athalia_interaction_matrix_1000.npz")
matrix = from_file("athalia_split_case.matrix.npz")
matrix.plot()
plt.show()

initial_path = [DirectedNode(contig, '+') for contig in range(matrix.n_contigs)]
splitted_path = bayesian_split(matrix, initial_path)
#matrix.plot_submatrix(70, 82)
#plt.show()

sys.exit()

joiner = IterativePathJoiner(matrix)
joiner.run(n_rounds=30)
path = joiner.get_final_path()
path_matrix = matrix.get_matrix_for_path(path, as_raw_matrix=False)

path_matrix.plot()
plt.show()

register_logging("logging")
contig_sizes = matrix._global_offset._contig_n_bins
initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]

#inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(matrix, max_bins=5000)
sums = inter_background.get_sums(inter_background.matrices.shape[0], inter_background.matrices.shape[1])
px.histogram(sums, title='Histogram of inter-contig sums').show()

#path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]
path = ContigPath.from_directed_nodes(path)
splitted_paths = split_using_inter_distribution(matrix, inter_background, path, threshold=0.00000005)

print(splitted_paths)
#to_file(path_matrix, "heatmap-contig200.png.matrix.2.npz")
matrix.plot()
plt.show()




