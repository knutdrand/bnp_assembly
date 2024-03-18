"""
Distance estimation between contig sides based on sparse interactiono matrix data
"""
import logging

import matspy
import plotly.express as px
import numpy as np
from tqdm import tqdm
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundMatrix, BackgroundInterMatrices
from bnp_assembly.util import get_all_possible_edges
from .plotting import px


def default_score_func(foreground_matrix, background_matrix):
    a = foreground_matrix.sum()
    b = background_matrix.sum() + 1
    return a / b


def median_difference_score(foreground_matrix, background_matrix):
    return np.median(foreground_matrix) / np.median(background_matrix)
    # scores = foreground_matrix.toarray() / (background_matrix.toarray() + 0.000001)
    return np.median(scores)


def get_distance_matrix_from_sparse_interaction_matrix(interactions: SparseInteractionMatrix,
                                                       background_interactions: BackgroundMatrix,
                                                       score_func: callable = None,
                                                       use_clipping_information=None) -> DirectedDistanceMatrix:
    all_edges = get_all_possible_edges(interactions.n_contigs)
    distances = DirectedDistanceMatrix(interactions.n_contigs)
    background = background_interactions.matrix
    # NB: Background is oriented away from the diagonal, shortest distance in lower left corner
    background = background[::-1, ]
    # matspy.spy(background)

    logging.info(f"Calculating distance matrix for all edges")
    for edge in tqdm(all_edges, total=len(distances) ** 2):
        if edge.from_node_side.node_id == edge.to_node_side.node_id:
            continue
        edge_submatrix = interactions.get_edge_interaction_matrix(edge, orient_according_to_nearest_interaction=True)

        # limit to maximum size of background matrix
        if score_func is None:
            score_func = default_score_func

        maxdist = 20
        maxdist = min(maxdist, background.shape[0])
        x_size, y_size = edge_submatrix.shape
        x_size = min(x_size // 2, maxdist)
        y_size = min(y_size // 2, maxdist)
        background_to_use = background[:x_size, :y_size]
        foreground = edge_submatrix[:x_size, :y_size]

        if use_clipping_information:
            # check if any of the nodesides are clipped, offset part of background we use to match clipping
            pass

        assert background_to_use.shape == foreground.shape
        score = score_func(foreground, background_to_use)
        distances[edge] = score

        if edge.to_node_side.node_id == edge.from_node_side.node_id + 1:
            px(name='distance').imshow(foreground.toarray(), title=f'Foreground {edge}')
            px(name='distance').imshow(background_to_use.toarray(), title=f'Background {edge}')

    return distances


def get_distance_matrix_from_sparse_interaction_matrix_as_p_values(interactions: SparseInteractionMatrix,
                                                                   background: BackgroundInterMatrices) -> DirectedDistanceMatrix:
    all_edges = get_all_possible_edges(interactions.n_contigs)
    distances = DirectedDistanceMatrix(interactions.n_contigs)

    logging.info(f"Calculating distance matrix for all edges")

    for edge in tqdm(all_edges, total=len(distances) ** 2):
        if edge.from_node_side.node_id == edge.to_node_side.node_id:
            continue
        edge_submatrix = interactions.get_edge_interaction_matrix(edge, orient_according_to_nearest_interaction=True)

        # limit to maximum size of background matrix
        maxdist = 200
        maxdist = min(maxdist, background.matrices.shape[1])

        x_size, y_size = edge_submatrix.shape
        x_size = min(x_size // 2, maxdist)
        y_size = min(y_size // 2, maxdist)
        foreground = edge_submatrix[:x_size, :y_size]

        foreground_sum = np.sum(foreground)
        score = background.get_percentile2(y_size, x_size, foreground_sum)
        assert score >= 0, score
        score = np.log(score + 0.0001)
        distances[edge] = score

    return distances


class DistanceFinder(EdgeDistanceFinder):
    def __init__(self, score_func: callable = None, contig_clips=None):
        self.score_func = score_func
        self.contig_clips = contig_clips

    def __call__(self, interactions: SparseInteractionMatrix, effective_contig_sizes=None) -> DirectedDistanceMatrix:
        background = BackgroundMatrix.from_sparse_interaction_matrix(interactions)
        dists = get_distance_matrix_from_sparse_interaction_matrix(interactions, background, self.score_func,
                                                                   use_clipping_information=self.contig_clips)
        dists.invert()
        return dists


class DistanceFinder2(EdgeDistanceFinder):
    def __init__(self):
        pass

    def __call__(self, interactions: SparseInteractionMatrix, effective_contig_sizes=None) -> DirectedDistanceMatrix:
        background = BackgroundInterMatrices.from_sparse_interaction_matrix(interactions)
        #background.plot()
        dists = get_distance_matrix_from_sparse_interaction_matrix_as_p_values(interactions, background)
        #dists.invert()
        return dists
