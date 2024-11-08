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
import numba
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
    def sampling_function(matrix, n_samples, max_bins, dynamic_sampling, dynamic_sampling_lowest_bin):
        return get_inter_background_means_stds(matrix, n_samples=n_samples, max_bins=max_bins,
                                               dynamic_sampling=dynamic_sampling, dynamic_sampling_lowest_bin=dynamic_sampling_lowest_bin)

    return get_background_means_stds_multires(interaction_matrix, sampling_function, n_samples, max_bins)

    n_iterations = 10
    inter_background_means = None
    inter_background_stds = None
    for i in range(n_iterations):

        #logging.info(f"Sampling {n_samples} reads for inter background with max bins {max_bins}")
        #b = get_intra_distance_background(interaction_matrix, n_samples=n_samples, max_bins=max_bins)
        next_bin_end = max_bins // 2
        if next_bin_end < 10:
            next_bin_end = 0
        means, stds = get_inter_background_means_stds(interaction_matrix, n_samples=n_samples, max_bins=max_bins,
                                                      dynamic_sampling=True, dynamic_sampling_lowest_bin=next_bin_end)
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
    background_sums = background.matrices.cumsum(axis=1).cumsum(axis=2)
    return background_sums



def get_background_means_stds_approximation(interactions, max_bins=1000):
    """Approximates fast using average number per bin"""
    rows, cols = np.nonzero(interactions.sparse_matrix)
    mean_nonzero = np.mean(interactions.sparse_matrix.data)
    prob_nonzero = len(rows) / (interactions.sparse_matrix.shape[0]**2)
    mean_value = np.sum(interactions.sparse_matrix) / (interactions.sparse_matrix.shape[0]**2)
    median_value = np.median(interactions.sparse_matrix.data)
    logging.info(f"Mean nonzero: {mean_nonzero}, median nonzero: {median_value}, prob nonzero: {prob_nonzero}")
    #logging.info(f"Mean value: {mean_value}")

    row_indices = np.arange(1, max_bins+1).reshape(-1, 1)
    col_indices = np.arange(1, max_bins+1)
    matrix = row_indices * col_indices
    matrix = matrix.astype(float)
    matrix *= (median_value*prob_nonzero)
    #matrix *= mean_value
    return matrix*1.5, matrix/2



def get_background_mean_stds(interactions, sampling_function, n_samples=1000, max_bins=1000,
                             dynamic_sampling=False, dynamic_sampling_lowest_bin=0, distance_weight_func=None):
    matrices = (matrix for matrix in
                sampling_function(interactions, n_samples=n_samples, max_bins=max_bins))

    w = Welford()
    n_nonzero_cols = None
    n_nonzero_rows = None
    for i, matrix in enumerate(matrices):
        if distance_weight_func is not None:
            # distance is row + col index
            distance = np.arange(matrix.shape[0]) + np.arange(matrix.shape[1]).reshape(-1, 1)
            weights = distance_weight_func(distance)
            matrix *= weights

        sums = matrix.cumsum(axis=0).cumsum(axis=1)
        w.add(sums)
        if dynamic_sampling:
            if n_nonzero_cols is None:
                n_nonzero_rows = np.zeros(matrix.shape[0])
                n_nonzero_cols = np.zeros(matrix.shape[1])
            dynamic_sampling_lowest_bin = min(dynamic_sampling_lowest_bin, matrix.shape[0] - 1)
            n_nonzero_rows += sums[:, dynamic_sampling_lowest_bin] > 0
            n_nonzero_cols += sums[dynamic_sampling_lowest_bin, :] > 0

            if np.all(n_nonzero_rows > 25) and np.all(n_nonzero_cols > 25):
                logging.info(
                    f"Stopping sampling after {i} samples because all rows and columns from {dynamic_sampling_lowest_bin} have enough data")
                break
    mean = w.mean
    variance = w.var_p
    return mean, np.sqrt(variance)


def get_inter_background_means_stds(interactions, n_samples=1000, max_bins=1000, dynamic_sampling=False, dynamic_sampling_lowest_bin=0, weight_func=None):
    """
    if dynamic_sampling, will sample until all values at row/column lowest_bin have enough data
    """
    return get_background_mean_stds(interactions, sample_inter_matrices, n_samples, max_bins, dynamic_sampling, dynamic_sampling_lowest_bin, distance_weight_func=weight_func)



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


def get_background_fixed_distance_mean_stds(interactions, n_samples=1000, max_bins=1000, distance_type="close", dynamic_sampling=False, dynamic_sampling_lowest_bin=0, weight_func=None):
    return get_background_mean_stds(interactions, lambda interactions, n_samples, max_bins: sample_with_fixed_distance_inside_big_contigs(interactions, max_bins, n_samples, distance_type), n_samples, max_bins,
                                    dynamic_sampling, dynamic_sampling_lowest_bin, distance_weight_func=weight_func)


def get_inter_as_mix_between_inside_outside(matrix, n_samples, max_bins, ratio=0.5, distance_type="far",
                                            dynamic_sampling=False, dynamic_sampling_lowest_bin=0, weight_func=None):
    #inter_inside = get_background_fixed_distance(matrix, n_samples, max_bins, distance_type)
    inter_inside_mean, inter_inside_std = get_background_fixed_distance_mean_stds(matrix, n_samples, max_bins,
                                                                                  distance_type, dynamic_sampling,
                                                                                  dynamic_sampling_lowest_bin, weight_func=weight_func)
    #return inter_inside
    #inter_inside2 = get_background_fixed_distance(matrix, n_samples, max_bins, "close")
    #inter_inside = np.concatenate([inter_inside, inter_inside2], axis=0)
    #inter_inside_mean = inter_inside.mean(axis=0)

    #inter_outside = get_inter_background(matrix, n_samples, inter_inside.shape[1])
    # don't need to sample more than inside here, this is only used for getting a mean to adjust for
    n_samples = 20
    inter_outside_mean, inter_outside_std = get_inter_background_means_stds(matrix, n_samples, inter_inside_mean.shape[0],
                                                                            dynamic_sampling, dynamic_sampling_lowest_bin,
                                                                            weight_func=weight_func)
    #inter_outside_mean = inter_outside.mean(axis=0)


    # mix between inside and outside (middle between)
    #inter = inter_inside - (inter_inside - inter_outside) * ratio
    means = inter_inside_mean #inter_inside.mean(axis=0)
    means = means - (means - inter_outside_mean) * ratio
    stds = inter_inside_std  #inter_inside.std(axis=0)
    #stds = inter_inside_std - (inter_inside_std - inter_outside_std) * ratio
    #stds = (inter_inside_std + inter_outside_std) / 2
    return means, stds


def get_intra_as_mix(matrix, n_samples, max_bins):
    assert False, "not to be used"
    intra_closest = get_background_fixed_distance(matrix, n_samples, max_bins, "closest")
    intra_close = get_background_fixed_distance(matrix, n_samples, max_bins, "close")
    intra = np.concatenate([intra_closest, intra_close], axis=0)
    return intra

def sample_intra_from_close(interaction_matrix, n_samples, max_bins):
    return sample_with_fixed_distance_inside_big_contigs(interaction_matrix, max_bins, n_samples, "close")

def sample_intra_from_close_and_closest(interaction_matrix, n_samples, max_bins):
    return sample_with_fixed_distance_inside_big_contigs(interaction_matrix, max_bins, n_samples, "close_closest")


def sample_intra_from_far(interaction_matrix, n_samples, max_bins):
    return sample_with_fixed_distance_inside_big_contigs(interaction_matrix, max_bins, n_samples, "far")


def get_intra_as_mix_means_stds(interaction_matrix, n_samples, max_bins, func=sample_intra_from_close_and_closest, distance_weight_func=None):
    def sampling_function(matrix, n_samples, max_bins, dynamic_sampling, dynamic_sampling_lowest_bin):
        return get_background_mean_stds(matrix, func, n_samples, max_bins, dynamic_sampling, dynamic_sampling_lowest_bin, distance_weight_func=distance_weight_func)
    return get_background_means_stds_multires(interaction_matrix, sampling_function, n_samples, max_bins)



def get_background_means_stds_multires(matrix, sampling_function, n_samples, max_bins):
    # uses multiple resolution to get enough data for close interactions
    n_resolutions = 10
    all_means = None
    all_stds = None
    #for n_samples, max_bins in resolutions:
    for i in range(n_resolutions):
        #logging.info(f"Sampling at resolution {max_bins} with {n_samples} (sample iteration {i})")
        next_lowest_bin = max_bins // 2
        if next_lowest_bin < 10:
            next_lowest_bin = 0

        means, stds = sampling_function(matrix, n_samples=n_samples, max_bins=max_bins,
                                                             dynamic_sampling=True, dynamic_sampling_lowest_bin=next_lowest_bin)
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


def get_inter_as_mix_between_inside_outside_multires(matrix, n_samples, max_bins, ratio=0.5, distance_type="far", distance_weight_func=None):
    sampling_function = lambda matrix, n_samples, max_bins, dynamic_sampling, dynamic_sampling_lowest_bin: (
        get_inter_as_mix_between_inside_outside(matrix, n_samples=n_samples, max_bins=max_bins, ratio=ratio, distance_type=distance_type, dynamic_sampling=dynamic_sampling, dynamic_sampling_lowest_bin=dynamic_sampling_lowest_bin, weight_func=distance_weight_func))
    return get_background_means_stds_multires(matrix, sampling_function, n_samples, max_bins)


# helper-function for getting counts
@numba.jit(nopython=True)
def _get_counts(values, contig1, contig2, dist_from_edge1, dist_from_edge2, nodeside_sizes, out, out_sizes1, out_sizes2):

    #for row, col, value, c1, c2, d1, d2 in zip(rows, columns, values, contig1, contig2, dist_from_edge1, dist_from_edge2):
    for i in range(len(values)):
        value = values[i]
        c1 = contig1[i]
        c2 = contig2[i]
        d1 = dist_from_edge1[i]
        d2 = dist_from_edge2[i]

        # don't look longer than a factor of the size of the smallest nodeside
        max_dist = min(nodeside_sizes[c1], nodeside_sizes[c2]) * 3

        effective_size1 = min(nodeside_sizes[c1], max_dist)
        effective_size2 = min(nodeside_sizes[c2], max_dist)
        out_sizes1[c1, c2] = effective_size1  # size of nodeside 1 when compared against node size 2
        out_sizes2[c1, c2] = effective_size2

        if d1 > max_dist or d2 > max_dist:
            continue

        out[c1, c2] += value



def get_edge_counts_with_max_distance(interactions: SparseInteractionMatrix, max_distance: int, trim_ratio=0.0, weight_func=None, use_numba=True) ->\
        Tuple[DirectedDistanceMatrix, np.ndarray]:
    """
    Returns a DistanceMatrix that represents the number of reads between all edges up to max_distance
    """
    t0 = time.perf_counter()
    # trick is to make a lookup array of a position to an node sidej, where bins further than max
    # distance are masked out

    lookup = np.zeros(interactions.sparse_matrix.shape[0]) - 1
    n_nodesides = interactions.n_contigs * 2
    nodeside_sizes = np.zeros(n_nodesides, dtype=int)
    distance_from_edge_lookup = np.zeros(interactions.sparse_matrix.shape[0], dtype=int)

    for contig in range(interactions.n_contigs):
        contig_start = interactions._global_offset.contig_first_bin(contig)
        contig_end = interactions._global_offset.contig_last_bin(contig, inclusive=False)

        if trim_ratio > 0:
            contig_start = contig_start + int((contig_end - contig_start) * trim_ratio)
            contig_end = contig_end - int((contig_end - contig_start) * trim_ratio)
        #contig_start = contig_start + (contig_end - contig_start) // 10
        #contig_end = contig_end - (contig_end - contig_start) // 10


        local_max = min((contig_end - contig_start)//2, max_distance)
        nodeside_start = contig*2
        nodeside_end = nodeside_start + 1
        lookup[contig_start:contig_start+local_max] = nodeside_start
        lookup[contig_end-local_max:contig_end] = nodeside_end

        nodeside_sizes[nodeside_start] = local_max
        nodeside_sizes[nodeside_end] = local_max

        if weight_func is not None or True:
            distance_from_edge_lookup[contig_start:contig_start+local_max] = np.arange(local_max)
            distance_from_edge_lookup[contig_end-local_max:contig_end] = np.arange(local_max)[::-1]



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
    rows = rows[~mask]
    columns = columns[~mask]
    dist_from_edge1 = distance_from_edge_lookup[rows]
    dist_from_edge2 = distance_from_edge_lookup[columns]



    if use_numba:
        out = np.zeros((n_nodesides, n_nodesides), dtype=float)
        out_sizes1 = np.ones((n_nodesides, n_nodesides), dtype=int) # init sizes with 1, means size will be one if there are no reads for interaction, which is probably fine
        out_sizes2 = np.ones((n_nodesides, n_nodesides), dtype=int)
        _get_counts(values, contig1, contig2, dist_from_edge1, dist_from_edge2, nodeside_sizes, out, out_sizes1, out_sizes2)
        matrix = DirectedDistanceMatrix(interactions.n_contigs)
        matrix._matrix = out
        logging.info(f"Time to compute edge counts: {time.perf_counter() - t0:.2f}")
        return matrix, (out_sizes1, out_sizes2)

    total_dist = dist_from_edge1 + dist_from_edge2
    if weight_func is not None:
        weights = weight_func(total_dist)
        values *= weights
        logging.info("Used weight func when computing edge counts, weights: %s", weights)

    indexes = contig1*n_nodesides + contig2
    indexes = indexes.astype(int)
    out = np.bincount(indexes, weights=values, minlength=n_nodesides*n_nodesides).reshape(n_nodesides, n_nodesides)
    matrix = DirectedDistanceMatrix(interactions.n_contigs)
    matrix._matrix = out
    logging.info(f"Time to compute edge counts: {time.perf_counter() - t0:.2f}")
    return matrix, nodeside_sizes


def get_prob_of_edge_counts(background_means: np.ndarray, background_stds: np.ndarray,
                            edge_counts, nodeside_sizes,
                            func=scipy.stats.norm.logsf, logging_name=None, relative_distance=False):
    # get background of intra counts
    t0 = time.perf_counter()

    maxdist = background_means.shape[0] - 1
    means = background_means
    stds = background_stds

    #n_nodesides = interaction_matrix.n_contigs * 2
    if isinstance(nodeside_sizes, Tuple):
        # hack: possible to send in a tuple with already initialized sizes1, and sizes2
        sizes1 = nodeside_sizes[0].ravel()
        sizes2 = nodeside_sizes[1].ravel()
        assert len(edge_counts.ravel()) == len(sizes1)
        n_nodesides = edge_counts.shape[1]
        assert n_nodesides**2 == len(sizes1)
        logging.info(f"Sizes1/2 matrices already initialized. {n_nodesides} nodesides. Shape: {sizes1.shape}")
    else:
        n_nodesides = len(nodeside_sizes)
        sizes1 = np.repeat(nodeside_sizes, n_nodesides)  # sizes of first node side in each edge
        sizes2 = np.tile(nodeside_sizes, n_nodesides)  # sizes of second node side in each edge
        logging.info(f"Sizes1/2 matrices not already initialized. {n_nodesides} nodesides. Shape: {sizes1.shape}")
    logging.info(sizes1)
    logging.info(sizes2)
    assert np.all(sizes1 > 0)
    assert np.all(sizes2 > 0)

    if relative_distance:
        max_relative_distance = maxdist // (3-1)
        logging.info(f"Max relative distance is {max_relative_distance}")
        distance_factor = 0.25
        # offset distance to be relative to contig sizes
        # calcuate the new mean/stds dynamically by subtracting
        # uses the smallest size when computing offests
        offsets = (np.minimum(sizes1, sizes2) * distance_factor).astype(int)
        offsets = np.minimum(max_relative_distance, offsets)
        s1 = sizes1-1 + offsets
        s2 = sizes2-1 + offsets
        assert np.all(s1 < means.shape[0])
        assert np.all(s2 < means.shape[0])
        edge_means = means[s1, s2]
        edge_means -= means[offsets, s2]
        edge_means -= means[s1, offsets]
        edge_means += means[offsets, offsets]
        size_is_1 = (sizes1 == 1) | (sizes2 == 1)
        edge_means[size_is_1] = means[sizes1[size_is_1]-1, sizes2[size_is_1]-1]  # don't use offset when one size is 1, use original formula
        #assert np.all(edge_means > 0)
        # set stds to be a factor of means as approc
        edge_stds = edge_means / 5
        logging.info(sizes1)
        logging.info(sizes2)
        logging.info(f"Used relative distance. Offsets: {offsets}")
    else:
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
    edge_scores = edge_counts.ravel()
    assert len(edge_scores) == len(edge_means), (len(edge_scores), len(edge_means))
    t0 = time.perf_counter()
    if logging_name is not None:
        np.save(f"{logging_name}_edge_scores.npy", edge_scores)
        np.save(f"{logging_name}_edge_means.npy", edge_means)
        np.save(f"{logging_name}_edge_stds.npy", edge_stds)

    pmfs = func(edge_scores, loc=edge_means, scale=edge_stds).reshape((n_nodesides, n_nodesides))
    logging.debug(f"Done p values, logpdf time: {time.perf_counter() - t0:.2f}")
    #px(name="joining").imshow(pmfs, title="pmfs").show()

    logging.info(f"Time to compute prob of edge counts: {time.perf_counter() - t0:.2f}")

    return DirectedDistanceMatrix.from_matrix(pmfs)


def get_bayesian_edge_probs(interaction_matrix: SparseInteractionMatrix,
                            inter_background_means: np.ndarray,
                            inter_background_stds: np.ndarray,
                            intra_background_means: np.ndarray,
                            intra_background_stds: np.ndarray,
                            normalize=False,
                            logging_name=None,
                            distance_weight_func=None,
                            relative_distance=False  # let distance from diagonal be relative to contig sizes
                            ):
    """
    Computes prob of in a bayesian way by looking both at prob of reads given edge and not edge
    means and stds are np.arrays with means and stds of matrices of size of the same shape as each position
    """

    maxdist = min(inter_background_means.shape[0], intra_background_means.shape[0])-1  #, inter_background_means2.shape[0]) - 1
    if relative_distance:
        largest_relative_distance = maxdist // 3
        maxdist -= largest_relative_distance

    logging.info("Maxdist for inter/intra counts is %d", maxdist)
    edge_counts, nodeside_sizes = get_edge_counts_with_max_distance(interaction_matrix, maxdist, trim_ratio=0.0, weight_func=distance_weight_func, use_numba=True)

    #tprob_reads_given_same_chrom = get_prob_of_edge_counts(inter_background_means2, inter_background_stds2, edge_counts, nodeside_sizes)

    #plotly.express.imshow(edge_counts.data, title="Edge counts").show()
    if logging_name is not None:
        logging.info(f"Logging edge counts, shape is {edge_counts.data.shape}. Matrix shape is {interaction_matrix.n_contigs*2}")
        np.save(f"{logging_name}_edge_counts.npy", edge_counts.data)
        np.save(f"{logging_name}_nodeside_sizes.npy", nodeside_sizes)

    def prob_func(x, loc, scale):
        return scipy.stats.t.logsf(x, df=np.zeros_like(x)+5, loc=loc, scale=scale)

    prob_reads_given_not_edge = get_prob_of_edge_counts(inter_background_means, inter_background_stds, edge_counts.data,
                                                        nodeside_sizes,
                                                        #func=scipy.stats.norm.logsf,
                                                        func=prob_func,
                                                        logging_name=logging_name + "_inter",
                                                        relative_distance=False) # don't use relative distance on inter-counts, too little data and no point
    prob_reads_given_not_edge = prob_reads_given_not_edge.data

    #plotly.express.imshow(prob_reads_given_not_edge[0:1000, 0:1000], title="Prob reads given not edge before normalize").show()


    #plotly.express.imshow(prob_reads_given_not_edge[0:1000, 0:1000], title="Prob reads given not edge").show()

    assert not np.any(np.isnan(prob_reads_given_not_edge)), prob_reads_given_not_edge
    assert not np.any(np.isinf(prob_reads_given_not_edge)), prob_reads_given_not_edge

    def prob_func2(x, loc, scale):
        return scipy.stats.t.logcdf(x, df=np.zeros_like(x)+5, loc=loc, scale=scale)

    prob_reads_given_edge = get_prob_of_edge_counts(intra_background_means, intra_background_stds, edge_counts.data,
                                                    nodeside_sizes,
                                                    #func=scipy.stats.norm.logcdf,
                                                    func=prob_func2,
                                                    logging_name=logging_name + "_intra",
                                                    relative_distance=relative_distance)

    assert not np.any(np.isnan(prob_reads_given_edge.data)), prob_reads_given_edge.data
    assert not np.any(np.isinf(prob_reads_given_edge.data)), prob_reads_given_edge.data

    #plotly.express.imshow(prob_reads_given_edge.data[0:1000, 0:1000], title="Prob reads given edge").show()

    prob_reads_given_edge = prob_reads_given_edge.data

    if logging_name is not None:
        np.save(f"{logging_name}_prob_reads_given_edge_not_normalized.npy", prob_reads_given_edge)
        np.save(f"{logging_name}_prob_reads_given_not_edge_not_normalized.npy", prob_reads_given_not_edge)


    if normalize:
        factor = min(prob_reads_given_edge.min(), prob_reads_given_not_edge.min())
        logging.info(f"Normalizing with {factor}")
        prob_reads_given_not_edge /= -factor

        logging.info(f"Normalizing with min {factor}")
        prob_reads_given_edge /= -factor

    if logging_name is not None:
        np.save(f"{logging_name}_prob_reads_given_edge.npy", prob_reads_given_edge)
        np.save(f"{logging_name}_prob_reads_given_not_edge.npy", prob_reads_given_not_edge)

    #prob_reads_given_same_chrom = prob_reads_given_same_chrom.data
    #prob_reads_given_not_edge = 1-prob_reads_given_not_edge.data


    #plt.imshow(prob_reads_given_edge)
    #plt.figure()
    #plt.imshow(prob_reads_given_not_edge)
    #plt.show()

    n_nodes = interaction_matrix.n_contigs * 2
    #prior_edge = np.log(1/n_nodes)
    prior_same_chrom = np.log(0.1)
    #prior_not_edge = np.log(1-(1/n_nodes))
    prior_edge = np.log(0.5)
    prior_not_edge = np.log(0.5)

    t0 = time.perf_counter()
    prob_observed_and_edge = prior_edge + prob_reads_given_edge
    prob_observed_and_not_edge = prior_not_edge + prob_reads_given_not_edge

    prob_edge = (prob_observed_and_edge -
                 np.logaddexp(prob_observed_and_edge,
                              prob_observed_and_not_edge,
                              #prior_same_chrom + prob_reads_given_same_chrom
                            ))
    #prob_edge = prob_edge.reshape(prob_reads_given_edge.shape)
    if logging_name is not None:
        np.save(f"{logging_name}_prob_edge.npy", prob_edge)


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
    background = background.matrix[:, ::-1, :]  # flip because oriented towards diagonal originallyj
    background_sums = background.cumsum(axis=1).cumsum(axis=2)
    assert np.all(background_sums >= 0)
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

