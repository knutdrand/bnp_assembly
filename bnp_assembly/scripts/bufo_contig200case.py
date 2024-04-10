import logging
logging.basicConfig(level=logging.INFO)
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, InteractionDistancesAndWeights
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split
from shared_memory_wrapper import from_file, to_file
import matplotlib.pyplot as plt


matrix = from_file("heatmap-contig200.png.matrix.2.npz")
#matrix.plot()
#plt.show()

contig_sizes = matrix._global_offset._contig_n_bins
initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]

new_path, path_matrix = greedy_bayesian_join_and_split(matrix, do_splitting=False)

#to_file(path_matrix, "heatmap-contig200.png.matrix.2.npz")

path_matrix.plot()
plt.show()
