import sys
import logging
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.cli import register_logging
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split
from bnp_assembly.missing_data import find_contig_clips_from_interaction_matrix
from bnp_assembly.sparse_interaction_matrix import BackgroundInterMatrices

logging.basicConfig(level=logging.INFO)
import numpy as np
from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, \
    InteractionDistancesAndWeights, split_using_inter_distribution
from bnp_assembly.sparse_interaction_based_distance import get_inter_background, get_intra_background, \
    get_intra_distance_background
from shared_memory_wrapper import from_file
import plotly.express as px
import matplotlib.pyplot as plt

register_logging("./logging")

#matrix = from_file("../tests/interaction_matrix_bufo.npz")
#scaffolds = ScaffoldAlignments.from_agp("bufo_scaffolds.agp")

matrix = from_file("athalia_interaction_matrix_1000.npz")
contig_sizes = {i: size for i, size in enumerate(matrix.contig_sizes)}
contig_clips = find_contig_clips_from_interaction_matrix(contig_sizes, matrix, window_size=100)
matrix.trim_with_clips(contig_clips)

joiner = IterativePathJoiner(matrix)
joiner.run()

sys.exit()
scaffolds = ScaffoldAlignments.from_agp("athalia.agp")


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
