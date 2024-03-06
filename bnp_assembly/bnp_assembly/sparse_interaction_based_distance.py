"""
Distance estimation between contig sides based on sparse interactiono matrix data
"""
import logging
from tqdm import tqdm
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundMatrix
from bnp_assembly.util import get_all_possible_edges


def get_distance_matrix_from_sparse_interaction_matrix(interactions: SparseInteractionMatrix, background_interactions: BackgroundMatrix):
    all_edges = get_all_possible_edges(interactions.n_contigs)
    distances = DirectedDistanceMatrix(interactions.n_contigs)
    background = background_interactions.matrix
    logging.info(f"Calculating distance matrix for all edges")
    for edge in tqdm(all_edges, total=len(distances)**2):
        if edge.from_node_side.node_id == edge.to_node_side.node_id:
            continue
        edge_submatrix = interactions.get_edge_interaction_matrix(edge, orient_according_to_nearest_interaction=True)

        # limit to maximum size of background matrix
        edge_submatrix = edge_submatrix[:background.shape[0], :background.shape[1]]
        x_size, y_size = edge_submatrix.shape
        score = edge_submatrix.sum() / background[:x_size, :y_size].sum()
        distances[edge] = score

    return distances
