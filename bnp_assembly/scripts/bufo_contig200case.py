import sys
import numpy as np
import logging
import plotly
from bnp_assembly.bayesian_splitting import bayesian_split, get_intra_median_distribution, get_inter_median_distribution
from bnp_assembly.cli import register_logging
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.missing_data import find_contig_clips_from_interaction_matrix
from bnp_assembly.sparse_interaction_based_distance import get_intra_distance_background, get_background_fixed_distance, \
    get_inter_background, get_inter_as_mix_between_inside_outside, get_inter_as_mix_between_inside_outside_multires, \
    get_intra_as_mix, get_inter_background_means_stds, get_inter_background_means_std_using_multiple_resolutions, \
    get_intra_as_mix_means_stds
from bnp_assembly.sparse_interaction_matrix import BackgroundInterMatrices, filter_low_mappability
import plotly.express as px
logging.basicConfig(level=logging.INFO)
from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, \
    InteractionDistancesAndWeights, split_using_inter_distribution
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split
from shared_memory_wrapper import from_file, to_file
import matplotlib.pyplot as plt

#matrix = from_file("../tests/interaction_matrix_bufo.npz")
#matrix = from_file("heatmap-contig4088.png.matrix.npz")
#matrix.plot(show_contigs=False)
#plt.show()
#intra = get_intra_as_mix(matrix, 100, 1000)
#inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(matrix, max_bins=5000)
#matrix = from_file("heatmap-contig1886.png.matrix.npz")
#matrix = from_file("scaffold_heatmap.png.matrix.5.npz")
#matrix = from_file("heatmap-contig50-1.png.matrix.npz")
matrix = from_file("thyatira_batis.npz")
#matrix = from_file("interaction_matrix_2000.pairs.npz")
# matrix.plot()
# plt.show()
matrix = filter_low_mappability(matrix)

matrix.plot()
plt.show()
#intra_background_means, intra_background_stds = get_intra_as_mix_means_stds(matrix, 2000, 5000)
#plt.imshow(intra_background_stds)
# matrix2.plot()
#plt.show()
#sys.exit()

#m, s = get_inter_background_means_std_using_multiple_resolutions(matrix, 50, 10000)
#m, s = get_inter_as_mix_between_inside_outside_multires(matrix, 50, 10000)
#plt.imshow(s)
#plt.show()
#inter_outside_mean, inter_outside_std = get_inter_background_means_stds(matrix, 20, 5000)
#print(inter_outside_mean[-1, -1], inter_outside_std[-1, -1])
#sys.exit()
#inter_means, inter_stds = get_inter_as_mix_between_inside_outside_multires(matrix, 50, 1000)
#plt.imshow(inter_means)
#inter = get_inter_as_mix_between_inside_outside(matrix, 5000, 60)
# plt.hist(intra[:, 16, 52]+0.1)
# plt.hist(inter[:, 16, 52])
# print(np.mean(intra[:, 16, 52]))
# print(np.std(intra[:, 16, 52]))
# plt.show()
# sys.exit()
#intra = get_background_fixed_distance(matrix, 500, 1000, "close")
#intra = get_intra_as_mix(matrix, 100, 1000)
#stds = intra.std(axis=0)
#print(np.sum(stds == 0))
#plt.imshow(stds)
#plt.show()
#inter = get_inter_as_mix_between_inside_outside(matrix, 50, 3000)
#plt.imshow(inter.mean(axis=0))
#plt.imshow(inter.std(axis=0))
#plt.show()
#sys.exit()
#matrix = from_file("athalia_interaction_matrix_1000.npz")
#matrix = from_file("athalia_case.matrix.npz")

"""
inter_outside = get_inter_background(matrix, 50, 1000)
inter_outside_mean = inter_outside.mean(axis=0)
inter_inside = get_background_fixed_distance(matrix, 50, 1000, "far")
inter_inside_mean = inter_inside.mean(axis=0)
inter = inter_inside - (inter_inside - inter_outside) * 0.8
plt.hist(inter_inside[:, 500, 500])
plt.hist(inter_outside[:, 500, 500])
plt.show()
sys.exit()
"""

initial_path = [DirectedNode(contig, '+') for contig in range(matrix.n_contigs)]
#initial_path = [DirectedNode(51, '-'), DirectedNode(50, '-'), DirectedNode(49, '-')]
#matrix = matrix.get_matrix_for_path(initial_path, as_raw_matrix=False)
#splitted_path = bayesian_split(matrix, initial_path, type='median')
#sys.exit()
#matrix.plot_submatrix(70, 82)
#plt.show()

#sys.exit()

joiner = IterativePathJoiner(matrix)
joiner.run(n_rounds=10)
path_matrix = joiner.current_interaction_matrix
#path_matrix.plot()
path = joiner.get_final_path()
#print("Length of final path", len(path))
path_matrix = matrix.get_matrix_for_path2(path, as_raw_matrix=False)
path_matrix.plot()
plt.show()
sys.exit()

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




