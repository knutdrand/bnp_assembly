import logging
from typing import List, Literal

import matspy
import scipy
from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.graph_objects import Edge, NodeSide
from bnp_assembly.sparse_interaction_based_distance import get_intra_distance_background, \
    get_inter_background_means_std_using_multiple_resolutions
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundInterMatrices, \
    sample_intra_matrices, sample_inter_matrices
import numpy as np
import plotly
from bnp_assembly.splitting import split_on_scores
from matplotlib import pyplot as plt

from .plotting import px



def rebin_matrix(matrix: np.ndarray, factor: int = 10):
    """
    Rebins matrix by summing up values in bins of size factor
    """
    new_shape = (matrix.shape[0] // factor + 1, matrix.shape[1] // factor + 1)

    cols, rows = matrix.nonzero()
    values = np.array(matrix[cols, rows]).ravel()
    cols = cols // factor
    rows = rows // factor
    new_matrix = np.bincount(cols * new_shape[1] + rows, weights=values, minlength=new_shape[0] * new_shape[1]).reshape(new_shape)
    return new_matrix


def median_from_matrix(matrix: np.ndarray):
    # rebin because most counts are zero
    # todo: this should probably be dynamic, e.g. rebin until most values are not zero anymore
    matrix = rebin_matrix(matrix, 500)
    #matspy.spy(matrix)
    #plt.show()
    values = matrix.flatten()
    return np.median(values)
    sorted_values = np.sort(values)
    #middle = sorted_values[len(sorted_values) // 10: len(sorted_values) - len(sorted_values) // 10]
    #print(middle)
    #return np.mean(middle)


def get_intra_median_distribution(interaction_matrix: SparseInteractionMatrix, window_size=10000):
    medians = []
    matrix_size = None
    for matrix in sample_intra_matrices(interaction_matrix, n_samples=150, max_bins=window_size, type="strong"):
        median = median_from_matrix(matrix)
        medians.append(median)
        matrix_size = matrix.shape[0]

    medians = np.array(medians)

    plotly.express.histogram(medians, title='Intra median distribution').show()
    plotly.express.histogram(np.log(medians+1), title='Intra median distribution log').show()
    return np.mean(medians), np.std(medians), matrix_size


def get_inter_median_distribution(interaction_matrix: SparseInteractionMatrix, window_size=10000):
    medians = []
    for matrix in sample_inter_matrices(interaction_matrix, max_bins=window_size, n_samples=150, keep_close=False):
        median = median_from_matrix(matrix)
        medians.append(median)

    medians = np.array(medians)

    plotly.express.histogram(medians, title='Inter median distribution').show()
    plotly.express.histogram(np.log(medians + 1), title='Inter median distribution log').show()
    return np.mean(medians), np.std(medians)


def bayesian_split(interaction_matrix: SparseInteractionMatrix, path: List[DirectedNode], threshold=1,
                   type: Literal["sum", "median"] = "sum"):
    """
    Splits by finding edges where ratio between prob of counts in some inter-window
    given edge and not edge is lower than some threshold.
    Filters so that there is only one split in each window of minimum chromosome size
    """
    max_size = 100000

    minimum_chromosome_size = min(max_size, interaction_matrix.sparse_matrix.shape[0] / 20)
    logging.info(f"Minimum chromosome size: {minimum_chromosome_size}")

    window_size = minimum_chromosome_size
    # backgrounds
    start_clip = 0
    if type == "sum":
        intra = get_intra_distance_background(interaction_matrix, n_samples=30, max_bins=window_size, type="strong", start_clip=start_clip)
        window_size = min(intra.shape[1], window_size)
        inter = BackgroundInterMatrices.from_sparse_interaction_matrix(interaction_matrix, max_bins=window_size, n_samples=30).matrices
        inter = inter[:, :-start_clip-1, start_clip:]
        inter = inter.cumsum(axis=1).cumsum(axis=2)
        #inter *= 5
        #inter = get_intra_distance_background(interaction_matrix, n_samples=100, max_bins=window_size, type="weak")
        #inter_means, inter_stds = get_inter_background_means_std_using_multiple_resolutions(interaction_matrix)

        intra_means = intra.mean(axis=0)
        inter_means = inter.mean(axis=0)
        intra_stds = intra.std(axis=0)
        inter_stds = inter.std(axis=0)
    else:
        intra_means, intra_stds, matrix_size = get_intra_median_distribution(interaction_matrix, window_size=window_size)
        inter_means, inter_stds = get_inter_median_distribution(interaction_matrix, window_size=matrix_size)

    m = interaction_matrix.sparse_matrix

    scores = np.zeros(interaction_matrix.n_contigs - 1)
    edge_scores = {}
    edge_counts = []

    for i, contig in enumerate(range(interaction_matrix.n_contigs-1)):
        node1 = contig
        node2 = contig + 1
        xstart = interaction_matrix.global_offset.contig_last_bin(node1)
        xend = min(m.shape[1], xstart + window_size)
        ystart = max(0, xstart - window_size)
        yend = xstart

        #edge_matrix = m[xstart:xend, ystart:yend]
        edge_matrix = m[ystart:yend, xstart:xend]
        edge_matrix = edge_matrix[:-start_clip-1, start_clip:]

        if type == "sum":
            count = edge_matrix.sum()

            x_size = edge_matrix.shape[1] - 1
            y_size = edge_matrix.shape[0] - 1
            intra_mean = intra_means[y_size, x_size]
            intra_std = intra_stds[y_size, x_size]
            inter_mean = inter_means[y_size, x_size]
            inter_std = inter_stds[y_size, x_size]

        else:
            count = median_from_matrix(edge_matrix)
            intra_mean = intra_means
            intra_std = intra_stds
            inter_mean = inter_means
            inter_std = inter_stds


        edge_counts.append(count)


        # cdf because prob of having more than

        if inter_std == 0:
            logging.warning(f"Inter std is 0 for {y_size}, {x_size}")
            inter_std = 0.000001

        prob_count_given_edge = scipy.stats.norm.logcdf(count, intra_mean, intra_std)
        prob_count_given_not_edge = scipy.stats.norm.logsf(count, inter_mean, inter_std)

        ratio = prob_count_given_edge - prob_count_given_not_edge

        contig1 = path[contig]
        contig2 = path[contig + 1]
        edge = Edge(contig1.right_side, contig2.left_side)
        edge_scores[edge] = ratio
        scores[contig] = ratio

        if i < 50:
            logging.info(f"Edge {edge} has score {ratio}. Count: {count}. "
                         f"Inter mean/std: {inter_mean}/{inter_std}. "
                         f"Intra mean/std: {intra_mean}/{intra_std}. "
                         f"Probs: {prob_count_given_edge}, {prob_count_given_not_edge}")


    # only split at edges with lowest score within window
    edge_positions = np.cumsum(interaction_matrix.contig_n_bins)[:-1]
    new_edge_scores = {}
    edge_index = 0

    # set threshold dynamiccaly based on median of scores
    score_values = list(edge_scores.values())
    median_score = np.median(score_values)
    px(name='splitting').array(score_values, title='Bayesian edge scores')
    logging.info("Median score: %f" % median_score)
    min_score = np.min(score_values)
    threshold = min(0, 0.1 * median_score)
    threshold = 0
    logging.info(f"Setting threshold dynamicaly to {threshold}")


    for edge, score in edge_scores.items():
        new_edge_scores[edge] = 0
        if score < threshold:
            # check that also lowest score among scores on edges with position less than window away
            indexes_to_check = np.where(np.abs(edge_positions[edge_index]-edge_positions) < window_size//2)[0]
            if score == np.min(scores[indexes_to_check]):
                new_edge_scores[edge] = 1
                #logging.info(f"Edge {edge} has score {score}, and is lowest in window")
            #else:
            #    logging.info(f"Edge {edge} has score {score}, but not lowest in window")

        edge_index += 1
    plotly.express.bar(y=list(edge_scores.values()), x=[str(edge) for edge in edge_scores.keys()], title='Bayesian edge scores').show()
    plotly.express.bar(y=edge_counts, x=[str(edge) for edge in edge_scores.keys()], title='Edge counts').show()
    splitted_paths = split_on_scores(ContigPath.from_directed_nodes(path), new_edge_scores, threshold=0.5, keep_over=False)
    return splitted_paths
