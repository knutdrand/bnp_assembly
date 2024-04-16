import logging
from bnp_assembly.cli import register_logging
from bnp_assembly.missing_data import find_contig_clips_from_interaction_matrix

logging.basicConfig(level=logging.INFO)
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, InteractionDistancesAndWeights
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split
from shared_memory_wrapper import from_file, to_file
import matplotlib.pyplot as plt


matrix = from_file("heatmap-contig200.png.matrix.3.npz")
matrix.plot()
plt.show()

register_logging("logging")
contig_sizes = matrix._global_offset._contig_n_bins
initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]

contig_sizes = {i: size for i, size in enumerate(matrix.contig_sizes)}

contig_clips = find_contig_clips_from_interaction_matrix(contig_sizes, matrix, window_size=100)
logging.info("Trimming interaction matrix with clips")
matrix.trim_with_clips(contig_clips)

new_path, path_matrix = greedy_bayesian_join_and_split(matrix, do_splitting=False)

#to_file(path_matrix, "heatmap-contig200.png.matrix.2.npz")

path_matrix.plot()
plt.show()
