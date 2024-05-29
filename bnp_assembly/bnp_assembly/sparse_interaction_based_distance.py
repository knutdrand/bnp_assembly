"""
Distance estimation between contig sides based on sparse interactiono matrix data
"""
import logging
import time
from typing import Tuple, Literal, Union

import matplotlib.pyplot as plt
import matspy
import plotly
import plotly.express as px
import numpy as np
import scipy
from tqdm import tqdm
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundMatrix, BackgroundInterMatrices, \
    BackgroundInterMatricesSingleBin, get_number_of_reads_between_all_contigs, \
    sample_with_fixed_distance_inside_big_contigs, sample_inter_matrices
from bnp_assembly.util import get_all_possible_edges
from welford import Welford

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
    background_sums = get_inter_background(interactions)
    return get_prob_of_edge_counts(background_sums, interactions)


def get_inter_background_means_std_using_multiple_resolutions(interaction_matrix, n_samples, max_bins):
    n_iterations = 7
    inter_background_means = None
    inter_background_stds = None
    for i in range(n_iterations):

        logging.info(f"Sampling {n_samples} reads for inter background with max bins {max_bins}")
        #b = get_intra_distance_background(interaction_matrix, n_samples=n_samples, max_bins=max_bins)
        means, stds = get_inter_background_means_stds(interaction_matrix, n_samples=n_samples, max_bins=max_bins)
        if inter_background_means is None:
            inter_background_means = means
            inter_background_stds = stds
        else:
            # overwrite the closer part with the one that has more samples (better estimate
            inter_background_means[:means.shape[0], :means.shape[1]] = means
            inter_background_stds[:means.shape[0], :means.shape[1]] = stds

        n_samples *= 4
        max_bins //= 2
        if max_bins < 10:
            break

    return inter_background_means, inter_background_stds


def get_inter_background(interactions, n_samples=1000, max_bins=1000):
    background = BackgroundInterMatrices.from_sparse_interaction_matrix(interactions, n_samples=n_samples, max_bins=max_bins)
    logging.info("Summing inter background")
    background_sums = background.matrices.cumsum(axis=1).cumsum(axis=2)
    logging.info("Done summing")
    return background_sums


def get_inter_background_means_stds(interactions, n_samples=1000, max_bins=1000):
    matrices = (matrix.toarray().astype(np.float32) for matrix in
                sample_inter_matrices(interactions, assumed_largest_chromosome_ratio_of_genome=1/10, n_samples=n_samples,
                                      max_bins=max_bins))
    w = Welford()
    for matrix in matrices:
        sums = matrix.cumsum(axis=0).cumsum(axis=1)
        w.add(sums)

    mean = w.mean
    variance = w.var_p
    return mean, np.sqrt(variance)


def get_intra_distance_background(interactions, n_samples=1000, max_bins=1000, type="weak", start_clip=0):
    background = BackgroundInterMatrices.weak_intra_interactions2(interactions, n_samples=n_samples, max_bins=max_bins, type=type)
    background.matrices = background.matrices[:, ::-1, :]  # flip because oriented towards diagonal originally
    background.matrices = background.matrices[:, :-start_clip-1, start_clip:]
    background_sums = background.matrices.cumsum(axis=1).cumsum(axis=2)
    return background_sums


def get_background_fixed_distance(interactions, n_samples=1000, max_bins=1000, distance_type="close", start_clip=0):
    matrices = np.array([matrix.toarray().astype(np.float32) for matrix in
                sample_with_fixed_distance_inside_big_contigs(interactions, max_bins, n_samples, distance_type)])
    matrices = matrices[:, ::-1, :]  # flip because oriented towards diagonal originally
    background_sums = matrices.cumsum(axis=1).cumsum(axis=2)
    return background_sums


def get_background_fixed_distance_mean_stds(interactions, n_samples=1000, max_bins=1000, distance_type="close", start_clip=0):
    matrices = (matrix.toarray().astype(np.float32) for matrix in
                sample_with_fixed_distance_inside_big_contigs(interactions, max_bins, n_samples, distance_type))
    w = Welford()
    for matrix in matrices:
        matrix = matrix[::-1, :]
        sums = matrix.cumsum(axis=0).cumsum(axis=1)
        w.add(sums)

    #matrices = matrices[:, ::-1, :]  # flip because oriented towards diagonal originally
    #background_sums = matrices.cumsum(axis=1).cumsum(axis=2)
    #return background_sums
    means = w.mean
    variance = w.var_p
    return means, np.sqrt(variance)


def get_inter_as_mix_between_inside_outside(matrix, n_samples, max_bins, ratio=0.5, distance_type="far"):
    #inter_inside = get_background_fixed_distance(matrix, n_samples, max_bins, distance_type)
    inter_inside_mean, inter_inside_std = get_background_fixed_distance_mean_stds(matrix, n_samples, max_bins, distance_type)
    #return inter_inside
    #inter_inside2 = get_background_fixed_distance(matrix, n_samples, max_bins, "close")
    #inter_inside = np.concatenate([inter_inside, inter_inside2], axis=0)
    #inter_inside_mean = inter_inside.mean(axis=0)

    #inter_outside = get_inter_background(matrix, n_samples, inter_inside.shape[1])
    inter_outside_mean, inter_outside_std = get_inter_background_means_stds(matrix, n_samples, inter_inside_mean.shape[0])
    #inter_outside_mean = inter_outside.mean(axis=0)


    # mix between inside and outside (middle between)
    #inter = inter_inside - (inter_inside - inter_outside) * ratio
    means = inter_inside_mean #inter_inside.mean(axis=0)
    means - (means - inter_outside_mean) * ratio
    stds = inter_inside_std  #inter_inside.std(axis=0)
    return means, stds


def get_intra_as_mix(matrix, n_samples, max_bins):
    intra_closest = get_background_fixed_distance(matrix, n_samples, max_bins, "closest")
    intra_close = get_background_fixed_distance(matrix, n_samples, max_bins, "close")
    intra = np.concatenate([intra_closest, intra_close], axis=0)
    return intra


def get_intra_as_mix_means_stds(interaction_matrix, n_samples, max_bins):
    # uses welford
    for dist_type in ["closest", "close"]:
        matrices = (matrix.toarray().astype(np.float32) for matrix in
                    sample_with_fixed_distance_inside_big_contigs(interaction_matrix, max_bins, n_samples, dist_type))
        w = Welford()
        for matrix in matrices:
            matrix = matrix[::-1, :]
            sums = matrix.cumsum(axis=0).cumsum(axis=1)
            w.add(sums)

    means = w.mean
    variance = w.var_p
    return means, np.sqrt(variance)


def get_inter_as_mix_between_inside_outside_multires(matrix, n_samples, max_bins, ratio=0.5, distance_type="far"):
    # uses multiple resolution to get enough data for close interactions
    n_resolutions = 7
    all_means = None
    all_stds = None
    #for n_samples, max_bins in resolutions:
    for i in range(n_resolutions):
        logging.info(f"Sampling at resolution {max_bins} with {n_samples}")
        means, stds = get_inter_as_mix_between_inside_outside(matrix, n_samples=n_samples, max_bins=max_bins, ratio=ratio, distance_type=distance_type)
        #means = b.mean(axis=0)
        #stds = b.std(axis=0)
        if all_means is None:
            all_means = means
            all_stds = stds
        else:
            # overwrite the closer part with the one that has more samples (better estimate
            all_means[:means.shape[0], :means.shape[1]] = means
            all_stds[:means.shape[0], :means.shape[1]] = stds

        n_samples *= 4
        max_bins //= 2

        if max_bins < 10:
            break

    #all_stds *= 5

    return all_means, all_stds


def get_edge_counts_with_max_distance(interactions: SparseInteractionMatrix, max_distance: int) ->\
        Tuple[DirectedDistanceMatrix, np.ndarray]:
    """
    Returns a DistanceMatrix that represents the number of reads between all edges up to max_distance
    """
    t0 = time.perf_counter()
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
    if isinstance(interactions.sparse_matrix, scipy.sparse.coo_matrix):
        values = interactions.sparse_matrix.data
    else:
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
    logging.info(f"Time to compute edge counts: {time.perf_counter() - t0:.2f}")
    return matrix, nodeside_sizes


def get_prob_of_edge_counts(background_means: np.ndarray, background_stds: np.ndarray,
                            edge_counts, nodeside_sizes,
                            func=scipy.stats.norm.logsf):
    # get background of intra counts

    maxdist = background_means.shape[0] - 1
    means = background_means
    stds = background_stds

    #n_nodesides = interaction_matrix.n_contigs * 2
    n_nodesides = len(nodeside_sizes)

    sizes1 = np.repeat(nodeside_sizes, n_nodesides)  # sizes of first node side in each edge
    sizes2 = np.tile(nodeside_sizes, n_nodesides)  # sizes of second node side in each edge

    edge_means = means[sizes1-1, sizes2-1]
    edge_stds = stds[sizes1-1, sizes2-1]

    if not np.all(edge_stds > 0):
        logging.warning(f"{np.sum(edge_stds == 0)} stds are 0.")
        logging.warning("Could be because too few reads")
        edge_stds[edge_stds == 0] = edge_means[edge_stds == 0] + 0.001
        # plotly.express.imshow(stds, title="stds").show()
        # plotly.express.imshow(means, title="means").show()
        # raise Exception("")

    n = n_nodesides

    # get p-values
    logging.info("Calculating p-values")
    edge_scores = edge_counts.data.ravel()
    assert len(edge_scores) == len(edge_means), (len(edge_scores), len(edge_means))
    t0 = time.perf_counter()
    pmfs = func(edge_scores, loc=edge_means, scale=edge_stds).reshape((n_nodesides, n_nodesides))
    logging.info(f"Done p values, logpdf time: {time.perf_counter() - t0:.2f}")
    #px(name="joining").imshow(pmfs, title="pmfs").show()

    return DirectedDistanceMatrix.from_matrix(pmfs)


def get_bayesian_edge_probs(interaction_matrix: SparseInteractionMatrix,
                            inter_background_means: np.ndarray,
                            inter_background_stds: np.ndarray,
                            intra_background_means: np.ndarray,
                            intra_background_stds: np.ndarray,
                            ):
    """
    Computes prob of in a bayesian way by looking both at prob of reads given edge and not edge
    means and stds are np.arrays with means and stds of matrices of size of the same shape as each position
    """

    maxdist = min(inter_background_means.shape[0], intra_background_means.shape[0])-1  #, inter_background_means2.shape[0]) - 1
    logging.info("Maxdist for inter/intra counts is %d", maxdist)
    edge_counts, nodeside_sizes = get_edge_counts_with_max_distance(interaction_matrix, maxdist)

    #prob_reads_given_same_chrom = get_prob_of_edge_counts(inter_background_means2, inter_background_stds2, edge_counts, nodeside_sizes)

    #plotly.express.imshow(edge_counts.data, title="Edge counts").show()
    prob_reads_given_not_edge = get_prob_of_edge_counts(inter_background_means, inter_background_stds, edge_counts,
                                                        nodeside_sizes,
                                                        func=scipy.stats.norm.logsf)
    prob_reads_given_not_edge = prob_reads_given_not_edge.data

    #plotly.express.imshow(prob_reads_given_not_edge[0:1000, 0:1000], title="Prob reads given not edge").show()

    assert not np.any(np.isnan(prob_reads_given_not_edge)), prob_reads_given_not_edge
    assert not np.any(np.isinf(prob_reads_given_not_edge)), prob_reads_given_not_edge

    prob_reads_given_edge = get_prob_of_edge_counts(intra_background_means, intra_background_stds, edge_counts,
                                                    nodeside_sizes,
                                                    func=scipy.stats.norm.logcdf)
    assert not np.any(np.isnan(prob_reads_given_edge.data)), prob_reads_given_edge.data
    assert not np.any(np.isinf(prob_reads_given_edge.data)), prob_reads_given_edge.data

    #plotly.express.imshow(prob_reads_given_edge.data[0:1000, 0:1000], title="Prob reads given edge").show()

    prob_reads_given_edge = prob_reads_given_edge.data
    #prob_reads_given_same_chrom = prob_reads_given_same_chrom.data
    #prob_reads_given_not_edge = 1-prob_reads_given_not_edge.data


    #plt.imshow(prob_reads_given_edge)
    #plt.figure()
    #plt.imshow(prob_reads_given_not_edge)
    #plt.show()

    n_nodes = interaction_matrix.n_contigs * 2
    prior_edge = np.log(1/n_nodes)
    prior_same_chrom = np.log(0.1)
    prior_not_edge = np.log(1-(1/n_nodes))

    t0 = time.perf_counter()
    prob_observed_and_edge = prior_edge + prob_reads_given_edge
    prob_edge = (prob_observed_and_edge -
                 np.logaddexp(prob_observed_and_edge,
                              prior_not_edge + prob_reads_given_not_edge,
                              #prior_same_chrom + prob_reads_given_same_chrom
                            ))
    logging.info(f"Time to compute probs: {time.perf_counter() - t0:.2f}")
    #prob_edge = prob_edge.reshape(prob_reads_given_edge.shape)


    #plt.figure()
    #plt.imshow(prob_edge)
    #plt.show()

    #prob_edge = -np.log(prob_edge)
    prob_edge = -prob_edge

    m = DirectedDistanceMatrix.from_matrix(prob_edge)
    m.fill_infs()

    prob_edge = m.data
    multi_zero = (prob_edge == 0).sum(axis=1) > 1
    if np.any(multi_zero):
        logging.warning(f"Multiple zeros in prob_edge for {np.sum(multi_zero)} nodesides")

    """
    zero_rows, zero_cols = np.where(prob_edge == 0)
    for row, col in zip(zero_rows, zero_cols):
        logging.warning(f"Nodeside sizes: {nodeside_sizes[row]}, {nodeside_sizes[col]}. "
                        f"Prob given edge: {prob_reads_given_edge[row, col]}, prob given not edge: {prob_reads_given_not_edge[row, col]}"
                        f"Edge count: {edge_counts.data[row, col]}"
                        f"Inter mean: {inter_background_means[nodeside_sizes[row]-1, nodeside_sizes[col]-1]}, std: {inter_background_stds[nodeside_sizes[row]-1, nodeside_sizes[col]-1]}"
                        f"Intra mean: {intra_background_means[nodeside_sizes[row]-1, nodeside_sizes[col]-1]}, std: {intra_background_stds[nodeside_sizes[row]-1, nodeside_sizes[col]-1]}")
    """

    #plotly.express.imshow(prob_edge[0:1000, 0:1000], title="Prob edge").show()
    return m



def get_intra_background(interaction_matrix, n_per_contig=10, max_contigs=10):
    background = BackgroundMatrix.from_sparse_interaction_matrix(interaction_matrix, create_stack=True, n_per_contig=n_per_contig,
                                                                 max_contigs=max_contigs)
    # get sum of background for all possible shapes
    logging.info("Doing cumsum")
    background = background.matrix[:, ::-1, :]  # flip because oriented towards diagonal originallyj
    background_sums = background.cumsum(axis=1).cumsum(axis=2)
    assert np.all(background_sums >= 0)
    logging.info("Cumsum done")
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

