from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
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
from bnp_assembly.location import LocationPair
from bnp_assembly.missing_data import find_contig_clips
from bnp_assembly.orientation_weighted_counter import OrientationWeightedCounter, OrientationWeightedCountesWithMissing
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.noise_distribution import NoiseDistribution
from bnp_assembly.plotting import px as px_func
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import log_config, DynamicHeatmapDistanceFinder, \
    get_dynamic_heatmap_config_with_even_bins, get_dynamic_heatmap_config_with_uniform_bin_sizes
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.scaffold_splitting.binned_bayes import NewSplitter
from bnp_assembly.splitting import YahsSplitter, split_on_scores
import logging

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

    split_paths =  numeric_split(numeric_input_data, contig_paths[0])
    return Scaffolds.from_contig_paths(split_paths, contig_name_translation)



def numeric_split(numeric_input_data: NumericInputData, path, bin_size=5000, max_distance=100000, threshold=0.2):
    assert isinstance(numeric_input_data.location_pairs, PairedReadStream), numeric_input_data.location_pairs
    return cluster_split(numeric_input_data, path)
    s = SplitterInterface(numeric_input_data.contig_dict, next(numeric_input_data.location_pairs), path,
                          max_distance=max_distance, bin_size=bin_size, threshold=threshold)
    return s.split()


def create_distance_matrix_from_reads(numeric_input_data: NumericInputData, edge_distance_finder: EdgeDistanceFinder,
                                      bin_size=1000) -> DirectedDistanceMatrix:
    contig_dict = numeric_input_data.contig_dict
    read_pairs = numeric_input_data.location_pairs
    contig_clips = find_contig_clips(bin_size, contig_dict, read_pairs)
    logger.info(f'contig_clis: {contig_clips}')
    new_contig_dict = {contig_id: end - start for contig_id, (start, end) in contig_clips.items()}
    assert all(v > 0 for v in new_contig_dict.values()), new_contig_dict
    del contig_dict
    clip_mapper = ClipMapper(contig_clips)

    # mapped_stream = clip_mapper.map_maybe_stream(next(read_pairs))
    # wrap clipped reads in PairedReadStream (need to have stream for some of the distance methods)
    new_read_stream = PairedReadStream((clip_mapper.map_maybe_stream(s) for s in read_pairs))
    distance_matrix = edge_distance_finder(new_read_stream, effective_contig_sizes=new_contig_dict)

    # adjusted_counts = adjust_counts_by_missing_data(counts, contig_dict, regions, cumulative_distribution, reads_per_bp, max_distance)
    # adjusted_counts = adjust_for_missing_data(counts, contig_dict, cumulative_distribution, bin_sizes)
    # forbes_obj = OrientationWeightedCountesWithMissing(contig_dict, next(read_pairs), cumulative_distribution)
    # distance_matrix = forbes_obj.get_distance_matrix()
    # distance_matrix = create_distance_matrix(len(new_contig_dict), counts, new_contig_dict)
    # distance_matrix.plot(name='forbes3')
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


def get_dynamic_heatmap_finder(numeric_input_data, cumulative_distribution, n_bins):
    max_distance_heatmaps = min(1000000, estimate_max_distance2(numeric_input_data.contig_dict.values()))
    max_gap_distance = min(5000000, (
        estimate_max_distance2(numeric_input_data.contig_dict.values()) * 2 - max_distance_heatmaps))
    heatmap_config = get_dynamic_heatmap_config_with_even_bins(cumulative_distribution,
                                                               n_bins=n_bins,
                                                               max_distance=max_distance_heatmaps)
    edge_distance_finder = DynamicHeatmapDistanceFinder(heatmap_config, max_gap_distance=max_gap_distance)
    return edge_distance_finder


def make_scaffold_numeric(numeric_input_data: NumericInputData, distance_measure='window', threshold=0.2,
                          bin_size=5000, splitting_method='poisson', max_distance=100000, **distance_kwargs) -> List[ContigPath]:
    assert isinstance(numeric_input_data.location_pairs, PairedReadStream), numeric_input_data.location_pairs
    if distance_measure == 'dynamic_heatmap':
        n_bins_heatmap_scoring = distance_kwargs["n_bins_heatmap_scoring"]
        joined = numeric_join(numeric_input_data, n_bins_heatmap_scoring)
        return numeric_split(numeric_input_data, joined, bin_size, max_distance, threshold)

    if distance_measure == 'forbes3' and splitting_method != 'poisson':
        cumulative_distribution = distance_dist(next(numeric_input_data.location_pairs), numeric_input_data.contig_dict)
        edge_distance_finder = ForbesDistanceFinder(numeric_input_data.contig_dict, cumulative_distribution,
                                                    max_distance)
        return default_make_scaffold(numeric_input_data, edge_distance_finder, threshold=threshold,
                                     max_distance=max_distance, bin_size=bin_size)

    contig_dict, read_pairs = numeric_input_data.contig_dict, next(numeric_input_data.location_pairs)
    if isinstance(read_pairs, List):
        read_pairs = read_pairs[0]  # some tests have a stream of read pairs

    # assert False
    px = px_func(name='joining')
    logging.info(f"Using splitting method {splitting_method} and distance measure {distance_measure}")
    if distance_measure == 'window':
        original_distance_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)
        split_matrix = original_distance_matrix
    elif distance_measure == 'forbes':
        pair_counts = get_pair_counts(contig_dict, read_pairs)
        node_side_counts = get_node_side_counts(pair_counts)
        d = {'length': [], 'node_side_count': [], 'node_side': []}
        for node_side in node_side_counts:
            d['length'].append(contig_dict[node_side.node_id])
            d['node_side_count'].append(node_side_counts[node_side])
            d['node_side'].append(str(node_side))
        pd.DataFrame(d).to_csv('node_side_counts.csv')
        DirectedDistanceMatrix.from_edge_dict(len(contig_dict), pair_counts).plot(name='pair counts')
        px.bar(x=[str(ns) for ns in node_side_counts.keys()], y=list(node_side_counts.values()),
               title='node side counts')
        original_distance_matrix = get_forbes_matrix(pair_counts, node_side_counts)
        original_distance_matrix.plot(name='forbes')
    elif distance_measure == 'forbes2':
        original_distance_matrix = OrientationWeightedCounter(contig_dict, read_pairs).get_distance_matrix()
        original_distance_matrix.plot(name='forbes2')
    elif distance_measure == 'forbes3':
        forbes_obj = OrientationWeightedCountesWithMissing(contig_dict, read_pairs, max_distance=max_distance)
        original_distance_matrix = forbes_obj.get_distance_matrix()
        original_distance_matrix.plot(name='forbes3')
    original_distance_matrix.inversion_plot('forbes2')
    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    path = join_all_contigs(distance_matrix)
    forbes_obj.plot_scores(forbes_obj.positions, forbes_obj.scores, edges=path.edges)
    if splitting_method == 'poisson':
        cumulative_distribution = CumulativeDistribution(
            distance_dist(read_pairs, contig_dict),
            p_noise=0.4,
            genome_size=sum(contig_dict.values()))
        # logging.info("Paths before splitting: %s" % paths)
        paths = split_contig_poisson(path, contig_dict, cumulative_distribution, threshold, original_distance_matrix,
                                     len(read_pairs.location_b))
    else:
        s = SplitterInterface(contig_dict, read_pairs, path, max_distance=100000, bin_size=5000)
        paths = s.split()
        # paths = [path]# split_contig(path, contig_dict, threshold*0.65, bin_size, read_pairs)
    return paths


def join_all_contigs(distance_matrix) -> ContigPath:
    mapping = None
    for _ in range(len(distance_matrix) // 2):
        paths = PathFinder(distance_matrix).run()
        print([path.directed_nodes for path in paths])
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            return ContigPath.from_node_sides(mapping.popitem()[1])
    assert False


def estimate_max_distance2(contig_sizes: Iterable[int]):
    """
    Finds a distance so contigs >  this distance cover at least 10% of total genome
    """
    sorted = np.sort(list(contig_sizes))[::-1]
    cumsum = np.cumsum(sorted)
    total_size = cumsum[-1]
    cutoff = np.searchsorted(cumsum, total_size // 10, side="right")
    return sorted[cutoff] // 8
