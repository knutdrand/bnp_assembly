import logging
from typing import List

import scipy
from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.graph_objects import Edge, NodeSide
from bnp_assembly.sparse_interaction_based_distance import get_intra_distance_background, \
    get_inter_background_means_std_using_multiple_resolutions
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundInterMatrices
import numpy as np
import plotly
from bnp_assembly.splitting import split_on_scores
from .plotting import px


def bayesian_split(interaction_matrix: SparseInteractionMatrix, path: List[DirectedNode], threshold=1):
    """
    Splits by finding edges where ratio between prob of counts in some inter-window
    given edge and not edge is lower than some threshold.
    Filters so that there is only one split in each window of minimum chromosome size
    """
    max_size = 5000

    minimum_chromosome_size = min(max_size, interaction_matrix.sparse_matrix.shape[0] / 20)
    logging.info(f"Minimum chromosome size: {minimum_chromosome_size}")

    window_size = minimum_chromosome_size
    # backgrounds
    intra = get_intra_distance_background(interaction_matrix, n_samples=100, max_bins=window_size, type="weak")
    window_size = min(intra.shape[1], window_size)
    inter = BackgroundInterMatrices.from_sparse_interaction_matrix(interaction_matrix, max_bins=window_size, n_samples=50).matrices
    inter = inter.cumsum(axis=1).cumsum(axis=2)
    #inter_means, inter_stds = get_inter_background_means_std_using_multiple_resolutions(interaction_matrix)

    intra_means = intra.mean(axis=0)
    inter_means = inter.mean(axis=0)
    intra_stds = intra.std(axis=0)
    inter_stds = inter.std(axis=0)

    m = interaction_matrix.sparse_matrix

    scores = np.zeros(interaction_matrix.n_contigs - 1)
    edge_scores = {}

    for contig in range(interaction_matrix.n_contigs-1):
        node1 = contig
        node2 = contig + 1
        xstart = interaction_matrix.global_offset.contig_last_bin(node1)
        xend = min(m.shape[1], xstart + window_size)
        ystart = max(0, xstart - window_size)
        yend = xstart

        edge_matrix = m[xstart:xend, ystart:yend]
        count = edge_matrix.sum()
        x_size = xend - xstart - 1
        y_size = yend - ystart - 1


        prob_count_given_edge = scipy.stats.norm.logcdf(count, intra_means[y_size, x_size], intra_stds[y_size, x_size])

        # cdf because prob of having more than

        inter_mean = inter_means[y_size, x_size]
        inter_std = inter_stds[y_size, x_size]
        if inter_std == 0:
            logging.warning(f"Inter std is 0 for {y_size}, {x_size}")
            inter_std = 0.000001
        prob_count_given_not_edge = scipy.stats.norm.logsf(count, inter_means[y_size, x_size], inter_std)


        ratio = prob_count_given_edge - prob_count_given_not_edge

        contig1 = path[contig]
        contig2 = path[contig + 1]
        edge = Edge(contig1.right_side, contig2.left_side)
        edge_scores[edge] = ratio
        scores[contig] = ratio

        logging.info(f"Edge {edge} has score {ratio}. Count: {count}. Inter mean/std: {inter_mean}/{inter_std}. "
                     f"Probs: {prob_count_given_edge}, {prob_count_given_not_edge}")


    # only split at edges with lowest score within window
    edge_positions = np.cumsum(interaction_matrix.contig_n_bins)[:-1]
    new_edge_scores = {}
    edge_index = 0
    for edge, score in edge_scores.items():
        new_edge_scores[edge] = 0
        if score < threshold:
            # check that also lowest score among scores on edges with position less than window away
            indexes_to_check = np.where(np.abs(edge_positions[edge_index]-edge_positions) < window_size//2)[0]
            if score == np.min(scores[indexes_to_check]):
                new_edge_scores[edge] = 1
                logging.info(f"Edge {edge} has score {score}, and is lowest in window")
            else:
                logging.info(f"Edge {edge} has score {score}, but not lowest in window")

        edge_index += 1
    plotly.express.bar(y=list(edge_scores.values()), x=[str(edge) for edge in edge_scores.keys()], title='Bayesian edge scores').show()
    splitted_paths = split_on_scores(ContigPath.from_directed_nodes(path), new_edge_scores, threshold=0.5, keep_over=False)
    return splitted_paths
