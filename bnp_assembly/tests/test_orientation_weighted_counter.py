from bnp_assembly.graph_objects import Edge
from bnp_assembly.orientation_weighted_counter import create_distance_matrix


def test_create_distance_matrix_acceptance():
    n_nodes = 5
    pair_counts = {Edge.from_numeric_index((i, j)): ((i+3*j) % 3)*10 for i in range(n_nodes*2) for j in range(n_nodes*2)}
    contig_dict = {i: i*20+1 for i in range(n_nodes)}
    create_distance_matrix(n_nodes, pair_counts, contig_dict, pseudo_count=0.1)
