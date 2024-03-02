import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shared_memory_wrapper import to_file

from bnp_assembly.contig_path_optimization import PathOptimizer, TotalDistancePathOptimizer, \
    flip_contigs_in_splitted_path, flip_contigs_in_splitted_path_path_optimizer, InteractionDistancesAndWeights, \
    LogProbSumOfReadDistancesDynamicScores
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, average_element_distance, \
    LogProbSumOfReadDistances, estimate_distance_pmf_from_sparse_matrix2
from numpy.testing import assert_array_equal
from scipy.stats import poisson

from bnp_assembly.clip_mapper import ClipMapper
from bnp_assembly.clustering import cluster_split
from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.distance_distribution import distance_dist
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.edge_distance_interface import EdgeDistanceFinder
from bnp_assembly.forbes_distance_calculation import ForbesDistanceFinder
from bnp_assembly.expected_edge_counts import ExpectedEdgeCounts, CumulativeDistribution
from bnp_assembly.forbes_score import get_pair_counts, get_node_side_counts, get_forbes_matrix
from bnp_assembly.input_data import FullInputData, NumericInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.max_distance import estimate_max_distance2
from bnp_assembly.missing_data import find_contig_clips, find_contig_clips_from_interaction_matrix
from bnp_assembly.orientation_weighted_counter import OrientationWeightedCounter, OrientationWeightedCountesWithMissing
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.noise_distribution import NoiseDistribution
from bnp_assembly.plotting import px as px_func
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmapDistanceFinder, \
    get_dynamic_heatmap_config_with_even_bins, get_dynamic_heatmap_config_with_uniform_bin_sizes
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.scaffold_splitting.binned_bayes import NewSplitter
from bnp_assembly.splitting import YahsSplitter, split_on_scores
import logging

from bnp_assembly.logprobsum_splitter import squares_split

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

    logger.info(f'contig_clips: {contig_clips}')
    new_contig_dict = {contig_id: end - start for contig_id, (start, end) in contig_clips.items()}
    logging.info("Using effective contig sizes after clipping: %s" % ('\n'.join(f"{k}:{v}" for k, v in new_contig_dict.items())))
    assert all(v > 0 for v in new_contig_dict.values()), new_contig_dict
    del contig_dict

    #clip_mapper = ClipMapper(contig_clips)
    #new_read_stream = PairedReadStream((clip_mapper.map_maybe_stream(s) for s in read_pairs))
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


def dynamic_heatmap_join_and_split(numeric_input_data: NumericInputData, n_bins_heatmap_scoring=20,
                                   split_threshold: float=10.0, interaction_matrix: SparseInteractionMatrix=None,
                                   interaction_matrix_clipping: SparseInteractionMatrix=None,
                                   cumulative_distribution=None):
    """Joins based on dynamic heatmaps and splits using the same heatmaps"""
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
    # note: interaction matrix is clipped inplace
    path = join_all_contigs(distance_matrix)

    directed_nodes = path.directed_nodes
    pickle.dump(directed_nodes, open("directed_nodes", "wb"))
    dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(interaction_matrix)

    distance_pmf = estimate_distance_pmf_from_sparse_matrix2(interaction_matrix).array
    distance_func = lambda dist: -distance_pmf[dist]
    path_contig_sizes = np.array([interaction_matrix.contig_n_bins[contig.node_id] for contig in directed_nodes])
    scorer = LogProbSumOfReadDistancesDynamicScores(directed_nodes.copy(), path_contig_sizes,
                                                    dists_weights, distance_func=distance_func)
    #new_directed_nodes = scorer.optimize_flippings()
    new_directed_nodes = scorer.optimize_positions()
    new_directed_nodes = scorer.optimize_flippings()

    logging.info(f"Optimized path:\nOld: {directed_nodes}\nNew: {new_directed_nodes}")
    path = ContigPath.from_directed_nodes(new_directed_nodes)

    if interaction_matrix.sparse_matrix.shape[1] < 1000000:
        interaction_matrix.plot_submatrix(0, interaction_matrix.n_contigs-1)
        new_matrix = interaction_matrix.get_matrix_for_path(new_directed_nodes, as_raw_matrix=False)
        new_matrix.plot_submatrix(0, interaction_matrix.n_contigs-1)
        plt.show()

    interaction_matrix.assert_is_symmetric()
    # split this path based on the scores from distance matrix
    edge_scores = distance_matrix.to_edge_dict()
    # must get edge scores in the same order as the path has edges
    edge_scores = {edge: score for edge, score in edge_scores.items() if edge in path.edges}
    edge_scores = {edge: score for edge, score in sorted(edge_scores.items(), key=lambda x: path.edges.index(x[0]))}
    logger.info(f"Edge dict: {edge_scores}")

    #split_threshold = min(split_threshold, max(edge_scores.values())-1)
    #logging.info(f"Adjusting split threshold to {split_threshold}")
    splitted_paths = split_on_scores(path, edge_scores, threshold=split_threshold, keep_over=False)

    #distance_pmf = estimate_distance_pmf_from_sparse_matrix2(interaction_matrix).array
    #logprobreaddist = LogProbSumOfReadDistances(distance_pmf)
    #evaluation_function = lambda x: -logprobreaddist(x)
    #splitted_paths = flip_contigs_in_splitted_path(interaction_matrix, splitted_paths)
    #splitted_paths = flip_contigs_in_splitted_path_path_optimizer(interaction_matrix, splitted_paths, evaluation_function)
    #to_file(interaction_matrix, "interaction_matrix_trimmed")
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


def make_scaffold_numeric(numeric_input_data: NumericInputData, distance_measure='window', threshold=0.2,
                          bin_size=5000, splitting_method='poisson', max_distance=100000, **distance_kwargs) -> List[
    ContigPath]:
    assert isinstance(numeric_input_data.location_pairs, PairedReadStream), numeric_input_data.location_pairs

    n_bins_heatmap_scoring = distance_kwargs["n_bins_heatmap_scoring"]
    return dynamic_heatmap_join_and_split(numeric_input_data,
                                          n_bins_heatmap_scoring=n_bins_heatmap_scoring,
                                          split_threshold=threshold,
                                          interaction_matrix=distance_kwargs.get("interaction_matrix", None),
                                          interaction_matrix_clipping=distance_kwargs.get("interaction_matrix_clipping", None),
                                          cumulative_distribution=distance_kwargs.get("cumulative_distribution", None),
                                          )



def join_all_contigs(distance_matrix) -> ContigPath:
    mapping = None
    for _ in range(len(distance_matrix) // 2):
        paths = PathFinder(distance_matrix).run()
        print([path.directed_nodes for path in paths])
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            return ContigPath.from_node_sides(mapping.popitem()[1])
    assert False


