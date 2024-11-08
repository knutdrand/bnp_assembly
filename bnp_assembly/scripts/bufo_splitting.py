import sys
import logging
from bionumpy import Genome
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.cli import register_logging
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split, get_contig_names_to_ids_translation
from bnp_assembly.missing_data import find_contig_clips_from_interaction_matrix
from bnp_assembly.sparse_interaction_matrix import BackgroundInterMatrices, weight_adjust_interaction_matrix

logging.basicConfig(level=logging.INFO)
import numpy as np
from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, \
    InteractionDistancesAndWeights, split_using_inter_distribution
from bnp_assembly.sparse_interaction_based_distance import get_inter_background, get_intra_background, \
    get_intra_distance_background, get_inter_as_mix_between_inside_outside_multires, \
    get_inter_background_means_std_using_multiple_resolutions, get_background_means_stds_approximation, \
    sample_intra_from_close, sample_intra_from_close_and_closest
from shared_memory_wrapper import from_file, to_file
import plotly.express as px
import matplotlib.pyplot as plt
from bnp_assembly.sparse_interaction_matrix import filter_low_mappability
from bnp_assembly.sparse_interaction_based_distance import get_intra_as_mix_means_stds

register_logging("./logging")

#matrix = from_file("../tests/interaction_matrix_bufo.npz")
#matrix = from_file("ilex.npz")
#matrix = from_file("meles_subset.npz")
matrix = from_file("heatmap-contig1504-20.png.matrix.npz")
means, stds = get_intra_as_mix_means_stds(matrix, func=sample_intra_from_close, n_samples=2000, max_bins=500)
means, stds = get_inter_as_mix_between_inside_outside_multires(matrix, 50, max_bins=500, ratio=0.99, distance_type="close")
plt.imshow(means)
plt.show()
sys.exit()

joiner = IterativePathJoiner(matrix, skip_init_distance_matrix=False)
#joiner.init_with_scaffold_alignments(scaffolds, contig_names_to_ids)
#joiner._interaction_matrix.plot()
joiner.run(n_rounds=30)

final_path = joiner.get_final_path()
path_matrix = matrix.get_matrix_for_path2(final_path, as_raw_matrix=False)
path_matrix.plot()
plt.show()
sys.exit()



contigs = scaffolds.contig_id.tolist()
directions = scaffolds.orientation.tolist()
numeric_ids = [int(contig.replace("contig", "")) for contig in contigs]

print("Contigs")
print(numeric_ids)

path = [DirectedNode(contig, dir) for contig, dir in zip(numeric_ids, directions)]
contig_path = ContigPath.from_directed_nodes(path)

path_matrix = matrix.get_matrix_for_path2(path, as_raw_matrix=False)


# splitting
dummy_path = [DirectedNode(i, '+') for i in range(matrix.n_contigs)]
inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix, max_bins=5000)
sums = inter_background.get_sums(inter_background.matrices.shape[0], inter_background.matrices.shape[1])
splitted_paths = split_using_inter_distribution(path_matrix,
                                                inter_background,
                                                ContigPath.from_directed_nodes(dummy_path),
                                                threshold=0.05)

path_matrix.plot()
plt.show()
sys.exit()

"""
contig_sizes = {i: size for i, size in enumerate(matrix.contig_sizes)}
contig_clips = find_contig_clips_from_interaction_matrix(contig_sizes, matrix, window_size=100)
matrix.trim_with_clips(contig_clips)

joiner = IterativePathJoiner(matrix)
joiner.run()
sys.exit()
"""




dummy_path = [DirectedNode(i, '+') for i in range(matrix.n_contigs)]


path_matrix.plot()
plt.show()

# splitting
inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix)
sums = inter_background.get_sums(inter_background.matrices.shape[0], inter_background.matrices.shape[1])
splitted_paths = split_using_inter_distribution(path_matrix,
                                                inter_background,
                                                ContigPath.from_directed_nodes(dummy_path),
                                                threshold=0.000000000005)

print("N scaffolds: ", len(splitted_paths))
