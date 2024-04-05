from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import LogProbSumOfReadDistancesDynamicScores, InteractionDistancesAndWeights
from shared_memory_wrapper import from_file


matrix = from_file("../tests/interaction_matrix_bufo.npz")
contig_sizes = matrix._global_offset._contig_n_bins
initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]

dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(matrix)
optimizer = LogProbSumOfReadDistancesDynamicScores(matrix, initial_path)
