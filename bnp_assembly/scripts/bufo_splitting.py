import sys
import logging
from bionumpy import Genome
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.cli import register_logging
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.make_scaffold import greedy_bayesian_join_and_split, get_contig_names_to_ids_translation
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

matrix = from_file("../tests/interaction_matrix_bufo.npz")
matrix = from_file("heatmap-contig1610.png.matrix.npz")
scaffolds = ScaffoldAlignments.from_agp("bufo_scaffolds.agp")
genome = Genome.from_file("bufo_contigs.fa.fai")

#matrix = from_file("athalia_interaction_matrix_1000.npz")
#scaffolds = ScaffoldAlignments.from_agp("athalia2.agp")
#genome = Genome.from_file("athalia_contigs.fa.fai")
"""
inter0 = get_inter_background(matrix, n_samples=1000, max_bins=500)
inter = get_intra_distance_background(matrix, n_samples=1000, max_bins=500, type="weak")
inter2 = np.concatenate([inter0, inter], axis=0)
intra = get_intra_distance_background(matrix, n_samples=1000, max_bins=500, type="strong")
i, j = 400, 400
px.histogram(inter0[:, i, j], nbins=10, title='inter0').show()
px.histogram(inter[:, i, j], nbins=10, title='inter').show()
px.histogram(inter2[:, i, j], nbins=10, title='inter2').show()
px.histogram(intra[:, i, j], nbins=60, title='intra').show()
print("Intra mean/std: ", np.mean(intra[:, i, j]), np.std(intra[:, i, j]))
print("Inter mean/std: ", np.mean(inter2[:, i, j]), np.std(inter2[:, i, j]))
sys.exit()
"""

#px.imshow(np.mean(b, axis=0)).show()

#sys.exit()

contig_names_to_ids = get_contig_names_to_ids_translation(genome)

#contig_sizes = {i: size for i, size in enumerate(matrix.contig_sizes)}
#contig_clips = find_contig_clips_from_interaction_matrix(contig_sizes, matrix, window_size=100)
#matrix.trim_with_clips(contig_clips)

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
