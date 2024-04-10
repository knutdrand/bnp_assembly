import pickle
from dataclasses import dataclass
import random
from typing import List
import numpy as np
import pandas as pd
from bnp_assembly.iterative_path_joining import IterativePathJoiner
from bnp_assembly.sparse_interaction_based_distance import DistanceFinder, DistanceFinder2, DistanceFinder3, \
    get_edge_counts_with_max_distance, get_prob_given_intra_background_for_edges, get_bayesian_edge_probs
from matplotlib import pyplot as plt

from bnp_assembly.contig_path_optimization import InteractionDistancesAndWeights, \
    LogProbSumOfReadDistancesDynamicScores, split_using_interaction_matrix
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, estimate_distance_pmf_from_sparse_matrix2
from scipy.stats import poisson

from bnp_assembly.clustering import cluster_split
from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.distance_distribution import distance_dist
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.expected_edge_counts import ExpectedEdgeCounts
from bnp_assembly.input_data import FullInputData, NumericInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.max_distance import estimate_max_distance2
from bnp_assembly.missing_data import find_contig_clips_from_interaction_matrix
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.noise_distribution import NoiseDistribution
from bnp_assembly.plotting import px as px_func
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmapDistanceFinder, \
    get_dynamic_heatmap_config_with_even_bins
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.scaffold_splitting.binned_bayes import NewSplitter
from bnp_assembly.splitting import YahsSplitter, split_on_scores
from shared_memory_wrapper import to_file
from .sparse_interaction_matrix import BackgroundInterMatrices

from .contig_path_optimization import optimize_splitted_path, split_using_inter_distribution
import logging

from bnp_assembly.logprobsum_splitter import squares_split
import plotly.express as px

logger = logging.getLogger(__name__)

PathFinder = nxPathFinder


def _split_contig(distance_matrix, path, T=-0.1):
    px_func('debug').histogram([distance_matrix[edge] for edge in path.edges if distance_matrix[edge] > -0.6],
                               nbins=15).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def split_contig_poisson(contig_path, contig_dict, cumulative_distribution, threshold, distance_matrix, n_read_pairs,
                         max_distance=100000):
    logging.info(f"Splitting with threshold {threshold}")
    expected_edge_counts = ExpectedEdgeCounts(contig_dict, cumulative_distribution, max_distance=max_distance)
    p_value_func = lambda observed, expected: poisson.cdf(observed, expected)
    log_prob_func = lambda observed, expected: poisson.logpmf(observed.astype(int), expected)
    noise_distribution = NoiseDistribution(contig_dict, distance_matrix, contig_path, max_distance=max_distance)
    px_func(name='splitting').histogram(noise_distribution.get_non_neighbour_scores(), nbins=100,
                                        title='non_neighbours')
    edge_scores = [np.exp(-distance_matrix[edge]) / noise_distribution.size_factor(edge) for edge in contig_path.edges]
    non_edge_probabilities = np.array([noise_distribution.edge_probability(edge) for edge in contig_path.edges])
    px_func(name='splitting').bar(edge_scores, title='noise_scores')
    px_func(name='splitting').bar(noise_distribution.cdf(edge_scores), title='noise_cumulative')

    observed = {edge: np.exp(-distance_matrix[edge]) for edge in contig_path.edges}
    expected = {edge: expected_edge_counts.get_expected_edge_count(edge) for edge in contig_path.edges}

    # ratio = sum(observed.values())/sum(expected.values())
    expected = {edge: expected[edge] * n_read_pairs for edge in contig_path.edges}
    px_func(name='splitting').scatter(y=list(expected.values()),
                                      x=[noise_distribution.rate(edge) for edge in contig_path.edges],
                                      color=[contig_dict[edge.from_node_side.node_id] + contig_dict[
                                          edge.from_node_side.node_id] for edge in contig_path.edges],
                                      size=[min(contig_dict[edge.from_node_side.node_id],
                                                contig_dict[edge.from_node_side.node_id]) for edge in
                                            contig_path.edges],
                                      title='rates')
    edge_probabililities = np.array([log_prob_func(observed[edge], expected[edge]) for edge in contig_path.edges])
    assert np.all(~np.isinf(edge_probabililities)), edge_probabililities
    table = [(str(edge), contig_dict[edge.from_node_side.node_id], contig_dict[edge.to_node_side.node_id],
              observed[edge], expected[edge])
             for edge in contig_path.edges]
    frame = pd.DataFrame(table, columns=['name', 'length_a', 'length_b', 'observed', 'expected'])
    frame['ratio'] = frame['observed'] / frame['expected']
    px_func(name='splitting').scatter(data_frame=frame, x='length_a', y='length_b', color='observed', size='expected',
                                      title='sizedep')
    px_func(name='splitting').scatter(data_frame=frame, x='observed', y='expected', title='observed vs expected')
    px_func(name='splitting').scatter(data_frame=frame, x='expected', y='ratio', title='ratio vs expected')
    ratio_func = lambda observed, expected: observed / expected * 1.5
    edge_scores = {
        edge: p_value_func(observed[edge], expected[edge])
        for edge in contig_path.edges}

    a = np.array(list(edge_scores.values()))
    ratio = np.exp(edge_probabililities - np.logaddexp(non_edge_probabilities, edge_probabililities))

    edge_scores = dict(zip(contig_path.edges, ratio))
    ratio_scores = {edge: ratio_func(observed[edge], expected[edge]) for edge in contig_path.edges}
    px_func(name='splitting').bar(y=ratio, x=[str(e) for e in contig_path.edges], title='posterior')
    px_func(name='splitting').bar(y=np.exp(edge_probabililities), x=[str(e) for e in contig_path.edges],
                                  title='edge-likelihood')
    px_func(name='splitting').bar(y=np.exp(non_edge_probabilities), x=[str(e) for e in contig_path.edges],
                                  title='non-edge-likelihood')
    return split_on_scores(contig_path, ratio_scores, threshold, keep_over=True)


def split_contig(contig_path, contig_dict, threshold, bin_size, locations_pair):
    YahsSplitter.matrix_class = NewSplitter
    return YahsSplitter(contig_dict, bin_size).split(contig_path, locations_pair, threshold=threshold)


def make_scaffold(input_data: FullInputData,
                  *args: object,
                  **kwargs: object) -> Scaffolds:
    contig_name_translation, numeric_input_data = get_numeric_input_data(input_data)
    contig_paths = make_scaffold_numeric(numeric_input_data, *args, **kwargs)
    scaffold = Scaffolds.from_contig_paths(contig_paths, contig_name_translation)
    return scaffold


def get_numeric_input_data(input_data):
    contig_sizes, contig_name_translation = get_numeric_contig_name_translation(input_data.contig_genome)
    reads = input_data.paired_read_stream
    if not isinstance(reads, PairedReadStream):
        reads = PairedReadStream(genomic_location_pair.get_numeric_locations() for genomic_location_pair in reads)
    numeric_input_data = NumericInputData(contig_sizes, reads)
    return contig_name_translation, numeric_input_data


def get_numeric_contig_name_translation(genome):
    encoding = genome.get_genome_context().encoding
    contig_dict = genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    contig_sizes = {int(encoding.encode(name).raw()): value for name, value in contig_dict.items()}
    return contig_sizes, translation_dict


@dataclass
class Scaffolder:
    def __call__(self, contig_dict, read_pairs_iter):
        cumulative_distribution = distance_dist(contig_dict, next(read_pairs_iter))
        distance_matrix = self.edge_scorer(contig_dict, next(read_pairs_iter),
                                           cumulative_distribution=cumulative_distribution)
        path = self.join_all(distance_matrix)
        return self.splitter(path, contig_dict, next(read_pairs_iter))


def default_make_scaffold(numeric_input_data, edge_distance_finder: EdgeDistanceFinder, threshold=0.2, bin_size=5000,
                          max_distance=100000) -> List[ContigPath]:
    distance_matrix = create_distance_matrix_from_reads(numeric_input_data, edge_distance_finder)
    path = join_all_contigs(distance_matrix)
    logger.info(f"Joined contigs: {path}")
    return numeric_split(numeric_input_data, path, bin_size, max_distance, threshold)


def split(input_data, scaffold: Scaffolds):
    contig_name_translation, numeric_input_data = get_numeric_input_data(input_data)
    contig_paths = scaffold.to_contig_paths(contig_name_translation)
    logger.info(f'Splitting path: {contig_paths[0]}')
    split_paths = numeric_split(numeric_input_data, contig_paths[0])
    logger.info(f'After split: {split_paths}')
    return Scaffolds.from_contig_paths(split_paths, contig_name_translation)


def numeric_split(numeric_input_data: NumericInputData, path, bin_size=5000, max_distance=100000, threshold=0.2):
    assert isinstance(numeric_input_data.location_pairs, PairedReadStream), numeric_input_data.location_pairs
    return squares_split(numeric_input_data, path)
    return cluster_split(numeric_input_data, path)
    s = SplitterInterface(numeric_input_data.contig_dict, next(numeric_input_data.location_pairs), path,
                          max_distance=max_distance, bin_size=bin_size, threshold=threshold)
    return s.split()


def create_distance_matrix_from_reads(numeric_input_data: NumericInputData, edge_distance_finder: EdgeDistanceFinder,
                                      bin_size=1000, use_clipping_to_adjust_distances: bool=False,
                                      interaction_matrix: SparseInteractionMatrix=None,
                                      interaction_matrix_clipping: SparseInteractionMatrix = None,

                                      ) -> DirectedDistanceMatrix:
    contig_dict = numeric_input_data.contig_dict
    original_contig_dict = contig_dict.copy()
    #read_pairs = numeric_input_data.location_pairs
    #contig_clips = find_contig_clips(bin_size, contig_dict, read_pairs)
    contig_clips = find_contig_clips_from_interaction_matrix(contig_dict, interaction_matrix_clipping, window_size=100)
    interaction_matrix.trim_with_clips(contig_clips)
    return None

    logger.info(f'contig_clips: {contig_clips}')
    new_contig_dict = {contig_id: end - start for contig_id, (start, end) in contig_clips.items()}
    logging.info("Using effective contig sizes after clipping: %s" % ('\n'.join(f"{k}:{v}" for k, v in new_contig_dict.items())))
    assert all(v > 0 for v in new_contig_dict.values()), new_contig_dict
    del contig_dict

    distance_matrix = edge_distance_finder(interaction_matrix, effective_contig_sizes=new_contig_dict)

    distance_matrix.plot(name="dynamic_heatmap_scores").show()
    distance_matrix.plot(name="dynamic_heatmap_scores_rr_and_ll", dirs='rrll').show()

    # adjust distances with clippings
    if use_clipping_to_adjust_distances:
        logging.info("Adjusting distance matrix with clipping")
        distance_matrix.adjust_with_clipping(contig_clips, original_contig_dict)
        distance_matrix.plot(name="dynamic_heatmap_scores_after_clipping_adjustment").show()

    return distance_matrix


def join(input_data, n_bins):
    contig_name_translation, numeric_input_data = get_numeric_input_data(input_data)
    contig_paths = [numeric_join(numeric_input_data, n_bins)]
    scaffold = Scaffolds.from_contig_paths(contig_paths, contig_name_translation)
    return scaffold


def numeric_join(numeric_input_data: NumericInputData, n_bins_heatmap_scoring=20):
    cumulative_distribution = distance_dist(next(numeric_input_data.location_pairs), numeric_input_data.contig_dict)

    edge_distance_finder = get_dynamic_heatmap_finder(numeric_input_data, cumulative_distribution,
                                                      n_bins_heatmap_scoring)
    distance_matrix = create_distance_matrix_from_reads(numeric_input_data,
                                                        edge_distance_finder)
    path = join_all_contigs(distance_matrix)
    return path


def greedy_bayesian_join_and_split(interaction_matrix: SparseInteractionMatrix = None, do_splitting=True):
    joiner = IterativePathJoiner(interaction_matrix)
    joiner.run()
    directed_nodes = joiner.get_final_path()
    path = ContigPath.from_directed_nodes(directed_nodes)
    path_matrix = interaction_matrix.get_matrix_for_path2(directed_nodes, as_raw_matrix=False)
    if not do_splitting:
        return path, path_matrix

    # splitting
    inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix)
    sums = inter_background.get_sums(inter_background.matrices.shape[0], inter_background.matrices.shape[1])

    px_func(name='splitting').histogram(sums, title='Histogram of inter-contig sums')
    #px_funx(name='splitting').histogram(np.sum(inter_background.matrices[:, :500, :500], axis=(1, 2)), title='inter matrices sums').show()
    splitted_paths = split_using_inter_distribution(path_matrix, inter_background, path, threshold=0.000000000005)

    return splitted_paths


def path_optimization_join_and_split(interaction_matrix: SparseInteractionMatrix = None,
                                     n_optimization_iterations=0,
                                     start_by_shuffling=True):
    """The path optimiztion method, tries to find optimal paths by sum of logprobs of read distances"""
    # start with some random order of contigs
    if interaction_matrix.n_contigs > 500:
        logging.info(f"Setting n optimization rounds to zero because too many contigs")
        n_optimization_iterations = 0

    directed_nodes = [DirectedNode(node_id, "+") for node_id in range(interaction_matrix.n_contigs)]
    #if start_by_shuffling:
    #   random.seed(0)
    #   random.shuffle(directed_nodes)
    #distance_matrix_counts = get_edge_counts_with_max_distance(interaction_matrix, 100)
    #distance_matrix_counts.plot(name="distance_matrix_counts").show()

    #distance_matrix = get_prob_given_intra_background_for_edges(interaction_matrix)

    distance_matrix = get_bayesian_edge_probs(interaction_matrix)
    px_func(name="main").array(distance_matrix.data, "bayesian_edge_probs")
    distance_matrix.plot(name="distance_matrix_p_values").show()
    distance_matrix.plot(name="distance_matrix_p_values_rrll", dirs='rrll').show()
    path = join_all_contigs(distance_matrix)
    directed_nodes = path.directed_nodes
    logging.info(f"Joined contigs: {path}")

    interaction_matrix.assert_is_symmetric()

    dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(interaction_matrix)
    set_to_flat_after_n_bins = int(1000000 / interaction_matrix.approx_global_bin_size())
    distance_pmf = estimate_distance_pmf_from_sparse_matrix2(interaction_matrix, set_to_flat_after_n_bins).array
    distance_func = lambda dist: -distance_pmf[dist]

    path_contig_sizes = np.array([interaction_matrix.contig_n_bins[contig.node_id] for contig in directed_nodes])
    scorer = LogProbSumOfReadDistancesDynamicScores(directed_nodes.copy(), path_contig_sizes,
                                                    dists_weights, distance_func=distance_func)
    prev_score = scorer.score()
    new_directed_nodes = directed_nodes
    logging.info(f"Will try to optimize path with {n_optimization_iterations} iterations")
    for i in range(n_optimization_iterations):
        scorer.optimize_by_moving_subpaths(interaction_matrix)
        scorer.optimize_positions()
        scorer.optimize_flippings()
        new_directed_nodes = scorer.get_path()
        logging.info(f"Path after iteration {i}: {new_directed_nodes}")

        #path = ContigPath.from_directed_nodes(new_directed_nodes)
        #path_matrix = interaction_matrix.get_matrix_for_path(new_directed_nodes, as_raw_matrix=False)
        #path_matrix.plot()
        px_func(name="main").matplotlib_figure(f"Interaction heatmap after {i+1} iterations")
        logging.info(f"Score after iteration: {scorer.score()}")
        if scorer.score() == prev_score:
            logging.info("No improvement in score after iteration %d, not trying more" % i)
            break
        prev_score = scorer.score()

    logging.info(f"Optimized path:\nOld: {directed_nodes}\nNew: {new_directed_nodes}")
    path = ContigPath.from_directed_nodes(new_directed_nodes)
    path_matrix = interaction_matrix.get_matrix_for_path(new_directed_nodes, as_raw_matrix=False)

    # splitting
    inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix)
    sums = inter_background.get_sums(inter_background.matrices.shape[0], inter_background.matrices.shape[1])
    px_func(name='splitting').histogram(sums, title='Histogram of inter-contig sums').show()
    #px_funx(name='splitting').histogram(np.sum(inter_background.matrices[:, :500, :500], axis=(1, 2)), title='inter matrices sums').show()
    splitted_paths = split_using_inter_distribution(path_matrix, inter_background, path, threshold=0.0005)

    # optimize the splited paths in the end
    if n_optimization_iterations > 0:
        splitted_paths = optimize_splitted_path(splitted_paths, interaction_matrix, dists_weights, distance_func)

    if interaction_matrix.sparse_matrix.shape[1] < 1000000000:
        interaction_matrix.plot_submatrix(0, interaction_matrix.n_contigs - 1)
        path_matrix.plot_submatrix(0, interaction_matrix.n_contigs - 1)
        px_func(name="main").matplotlib_figure("final_scaffold_heatmap")
        plt.show()

    return splitted_paths


def dynamic_heatmap_join_and_split(numeric_input_data: NumericInputData, n_bins_heatmap_scoring=20,
                                   split_threshold: float=10.0, interaction_matrix: SparseInteractionMatrix=None,
                                   interaction_matrix_clipping: SparseInteractionMatrix=None,
                                   cumulative_distribution=None):
    """Joins based on dynamic heatmaps and splits using the same heatmaps
    Thi method is outdated, kept for reference."""
    interaction_matrix.assert_is_symmetric()
    logging.info("Getting cumulative dist")
    if cumulative_distribution is None:
        cumulative_distribution = distance_dist(next(numeric_input_data.location_pairs), numeric_input_data.contig_dict)
    logging.info("Cumulative dist done")

    edge_distance_finder = get_dynamic_heatmap_finder(numeric_input_data, cumulative_distribution,
                                                      n_bins_heatmap_scoring)

    distance_matrix = create_distance_matrix_from_reads(numeric_input_data,
                                                        edge_distance_finder,
                                                        use_clipping_to_adjust_distances=False,
                                                        interaction_matrix=interaction_matrix,
                                                        interaction_matrix_clipping=interaction_matrix_clipping
                                                        )

    path = join_all_contigs(distance_matrix)
    directed_nodes = path.directed_nodes

    # note: interaction matrix is clipped inplace

    path_matrix = interaction_matrix.get_matrix_for_path(directed_nodes, as_raw_matrix=False)
    path_matrix.plot("Before splitting")
    plt.show()

    interaction_matrix.assert_is_symmetric()

    inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix, n_samples=200)
    splitted_paths = split_using_inter_distribution(path_matrix, inter_background, path, threshold=0.0005)

    if interaction_matrix.sparse_matrix.shape[1] < 1000000000:
        interaction_matrix.plot_submatrix(0, interaction_matrix.n_contigs - 1)
        path_matrix.plot_submatrix(0, interaction_matrix.n_contigs - 1)
        #plt.show()

    return splitted_paths
    # return path


def get_dynamic_heatmap_finder(numeric_input_data, cumulative_distribution, n_bins):
    max_distance_heatmaps = min(500000, estimate_max_distance2(numeric_input_data.contig_dict.values()))
    max_gap_distance = min(5000000, (
            estimate_max_distance2(numeric_input_data.contig_dict.values()) * 2 - max_distance_heatmaps))
    heatmap_config = get_dynamic_heatmap_config_with_even_bins(cumulative_distribution,
                                                               n_bins=n_bins,
                                                               max_distance=max_distance_heatmaps)
    #heatmap_config = get_dynamic_heatmap_config_with_uniform_bin_sizes(20, 200)
    edge_distance_finder = DynamicHeatmapDistanceFinder(heatmap_config, max_gap_distance=max_gap_distance)
    return edge_distance_finder


def make_scaffold_numeric(numeric_input_data: NumericInputData=None, distance_measure='window', threshold=0.2,
                          bin_size=5000, splitting_method='poisson', max_distance=100000, **distance_kwargs) -> List[
    ContigPath]:
    #assert isinstance(numeric_input_data.location_pairs, PairedReadStream), numeric_input_data.location_pairs

    # trim contigs, interaction matrix is clipped inplace
    interaction_matrix: SparseInteractionMatrix = distance_kwargs.get("interaction_matrix", None)
    interaction_matrix_clipping = distance_kwargs.get("interaction_matrix_clipping", None)

    contig_sizes = {i: size for i, size in enumerate(interaction_matrix.contig_sizes)}

    #contig_clips = find_contig_clips_from_interaction_matrix(contig_sizes, interaction_matrix_clipping, window_size=100)
    logging.info("Trimming interaction matrix with clips")
    #interaction_matrix.trim_with_clips(contig_clips)

    return greedy_bayesian_join_and_split(interaction_matrix)
    #return path_optimization_join_and_split(interaction_matrix=interaction_matrix)



def join_all_contigs(distance_matrix) -> ContigPath:
    mapping = None
    for i in range(len(distance_matrix) // 2):
        print("Iteration ", i)
        paths = PathFinder(distance_matrix).run()
        print([path.directed_nodes for path in paths])
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            return ContigPath.from_node_sides(mapping.popitem()[1])
    assert False


