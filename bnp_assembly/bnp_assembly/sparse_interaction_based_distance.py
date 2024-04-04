"""
Distance estimation between contig sides based on sparse interactiono matrix data
"""
import logging
from typing import Tuple, Literal, Union

import matplotlib.pyplot as plt
import matspy
import plotly.express as px
import numpy as np
import scipy
from tqdm import tqdm
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundMatrix, BackgroundInterMatrices, \
    BackgroundInterMatricesSingleBin, get_number_of_reads_between_all_contigs, \
    BackgroundInterMatricesMultipleBinsBackend
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
        continue


        # limit to maximum size of background matrix
        maxdist = 20000
        maxdist = min(maxdist, background.maxdist)

        x_size, y_size = edge_submatrix.shape
        x_size = min(x_size // 2, maxdist)
        y_size = min(y_size // 2, maxdist)
        foreground = edge_submatrix[:x_size, :y_size]

        foreground_sum = np.sum(foreground)
        #score = foreground_sum
        #score = background.get_percentile2(y_size, x_size, foreground_sum)
        score = background.cdf(y_size, x_size, foreground_sum)
        assert score >= 0, score
        assert not np.isnan(score), score
        score = np.log(1-score + 0.0000001)
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

        foreground = get_number_of_reads_between_all_contigs(interactions)
        plt.imshow(foreground)
        plt.show()

        #background = BackgroundInterMatrices.from_sparse_interaction_matrix(interactions)
        background = BackgroundInterMatricesSingleBin.from_sparse_interaction_matrix(interactions)
        #background.plot()
        dists = get_distance_matrix_from_sparse_interaction_matrix_as_p_values(interactions, background)
        #dists.invert()
        return dists


class DistanceFinder3(EdgeDistanceFinder):
    def __init__(self):
        pass

    def __call__(self, interactions: SparseInteractionMatrix, effective_contig_sizes=None) -> DirectedDistanceMatrix:

        """
        Computes p-valies of observed efficiently by looking at all reads between contigs
        """
        return get_prob_of_reads_given_not_edge(interactions, "cdf")


def get_prob_of_reads_given_not_edge(interactions, prob_func: Literal["cdf", "pmf", "logcdf", "logpdf"]="logcdf"):
    background = BackgroundInterMatrices.from_sparse_interaction_matrix(interactions, n_samples=1000)
    background_sums = background.matrices.cumsum(axis=1).cumsum(axis=2)
    return get_prob_of_edge_counts(background_sums, interactions)



def get_edge_counts_with_max_distance(interactions: SparseInteractionMatrix, max_distance: int) ->\
        Tuple[DirectedDistanceMatrix, np.ndarray]:
    """
    Returns a DistanceMatrix that represents the number of reads between all edges up to max_distance
    """


    # trick is to make a lookup array of a position to an node sidej, where bins further than max
    # distance are masked out

    logging.info("Making lookup")
    lookup = np.zeros(interactions.sparse_matrix.shape[0]) - 1
    n_nodesides = interactions.n_contigs * 2
    nodeside_sizes = np.zeros(n_nodesides, dtype=int)

    for contig in range(interactions.n_contigs):
        contig_start = interactions._global_offset.contig_first_bin(contig)
        contig_end = interactions._global_offset.contig_last_bin(contig, inclusive=False)
        local_max = min((contig_end - contig_start)//2, max_distance)
        nodeside_start = contig*2
        nodeside_end = nodeside_start + 1
        lookup[contig_start:contig_start+local_max] = nodeside_start
        lookup[contig_end-local_max:contig_end] = nodeside_end

        nodeside_sizes[nodeside_start] = local_max
        nodeside_sizes[nodeside_end] = local_max


    logging.info("Calculating matrix")
    rows, columns = interactions.sparse_matrix.nonzero()
    values = np.array(interactions.sparse_matrix[rows, columns]).ravel()

    contig1 = lookup[rows]
    contig2 = lookup[columns]

    mask = np.zeros_like(contig1, dtype=bool)
    mask[contig1 == -1] = True
    mask[contig2 == -1] = True
    contig1 = contig1[~mask].astype(int)
    contig2 = contig2[~mask].astype(int)
    values = values[~mask]

    indexes = contig1*n_nodesides + contig2
    indexes = indexes.astype(int)
    logging.info("Doing bincount")
    out = np.bincount(indexes, weights=values, minlength=n_nodesides*n_nodesides).reshape(n_nodesides, n_nodesides)

    logging.info("Making distance matrix")
    matrix = DirectedDistanceMatrix(interactions.n_contigs)
    matrix._matrix = out
    return matrix, nodeside_sizes


def get_prob_given_intra_background_for_edges(interaction_matrix: SparseInteractionMatrix):
    background_sums = get_intra_background_sums(interaction_matrix)
    return get_prob_of_edge_counts(background_sums, interaction_matrix)


def get_prob_of_edge_counts(background: np.ndarray, interaction_matrix: SparseInteractionMatrix):
    # get background of intra counts

    background_sums = background
    maxdist = background_sums.shape[1] - 1

    # get counts for all edges up to maxdist
    # todo: this call can either be cashed or done outside this function to speedup
    edge_counts, nodeside_sizes = get_edge_counts_with_max_distance(interaction_matrix, maxdist)

    #edge_counts.plot(name="edge_counts").show()
    #px(name="joining").imshow(edge_counts.data, title="edge_counts").show()

    logging.info("Getting means and stds")
    #np.save('background_sums.npy', background_sums)
    #plt.hist(background_sums[:, 4, 4])
    means = np.mean(background_sums, axis=0)
    px(name="joining").array(means, title="means")
    #plt.show()
    stds = np.std(background_sums, axis=0)
    if not np.all(stds > 0):
        logging.warning("Some stds are 0, cannot compute probs. Will fix by adding small number to those that are zero")
        logging.warning("Could be because too few reads")
        stds[stds == 0] = 0.0000001

    #px(name="joining").imshow(means, title="means").show()
    #px(name="joining").imshow(stds, title="stds").show()
    n_nodesides = interaction_matrix.n_contigs * 2

    sizes1 = np.repeat(nodeside_sizes, n_nodesides)  # sizes of first node side in each edge
    sizes2 = np.tile(nodeside_sizes, n_nodesides)  # sizes of second node side in each edge

    edge_means = means[sizes1-1, sizes2-1]
    edge_stds = stds[sizes1-1, sizes2-1]

    n = n_nodesides
    #px(name="joining").imshow(edge_means.reshape((n, n)), title="edge mean_nodesides").show()
    #px(name="joining").imshow(edge_stds.reshape((n, n)), title="edgestds").show()

    # get p-values
    logging.info("Calculating p-values")
    edge_scores = edge_counts.data.ravel()
    assert len(edge_scores) == len(edge_means), (len(edge_scores), len(edge_means))
    pmfs = scipy.stats.norm.logpdf(edge_scores, loc=edge_means, scale=edge_stds).reshape((n_nodesides, n_nodesides))
    #px(name="joining").imshow(pmfs, title="pmfs").show()

    return DirectedDistanceMatrix.from_matrix(pmfs)


def get_bayesian_edge_probs(interaction_matrix: SparseInteractionMatrix):
    """
    Computes prob of in a bayesian way by looking both at prob of reads given edge and not edge
    """
    prob_reads_given_not_edge = get_prob_of_reads_given_not_edge(interaction_matrix, "logpdf")
    assert not np.any(np.isnan(prob_reads_given_not_edge.data)), prob_reads_given_not_edge.data
    assert not np.any(np.isinf(prob_reads_given_not_edge.data)), prob_reads_given_not_edge.data

    prob_reads_given_edge = get_prob_given_intra_background_for_edges(interaction_matrix)
    prob_reads_given_edge.plot(name="prob_reads_given_edge").show()
    assert not np.any(np.isnan(prob_reads_given_edge.data)), prob_reads_given_edge.data
    assert not np.any(np.isinf(prob_reads_given_edge.data)), prob_reads_given_edge.data

    prob_reads_given_not_edge.plot(name="prob_reads_given_not_edge").show()

    prob_reads_given_edge = prob_reads_given_edge.data
    prob_reads_given_not_edge = prob_reads_given_not_edge.data
    #prob_reads_given_not_edge = 1-prob_reads_given_not_edge.data


    plt.imshow(prob_reads_given_edge)
    plt.figure()
    plt.imshow(prob_reads_given_not_edge)
    #plt.show()

    prior_edge = np.log(0.05)
    prior_not_edge = np.log(1-0.05)

    prob_observed_and_edge = prior_edge + prob_reads_given_edge
    prob_edge = (prob_observed_and_edge -
                 np.logaddexp(prob_observed_and_edge, prior_not_edge + prob_reads_given_not_edge))
    #prob_edge = prob_edge.reshape(prob_reads_given_edge.shape)

    #plt.figure()
    #plt.imshow(prob_edge)
    #plt.show()

    #prob_edge = -np.log(prob_edge)
    prob_edge = -prob_edge

    m = DirectedDistanceMatrix.from_matrix(prob_edge)
    m.fill_infs()
    return m



def get_intra_background_sums(interaction_matrix):
    background = BackgroundMatrix.from_sparse_interaction_matrix(interaction_matrix, create_stack=True, n_per_contig=5, max_contigs=10)
    # get sum of background for all possible shapes
    background = background.matrix[:, ::-1, :]  # flip because oriented towards diagonal originallyj
    background_sums = background.cumsum(axis=1).cumsum(axis=2)
    return background_sums


def get_negative_binom_params(mean, variance):
    n = mean**2 / (variance - mean)
    p = mean / variance
    return n, p
    #p = variance / mean
    #r = mean / p
    #return r, p

def get_negative_binomial_distribution(mean, variance):
    from scipy.stats import nbinom
    n, p = get_negative_binom_params(mean, variance)
    return nbinom(n, p)

