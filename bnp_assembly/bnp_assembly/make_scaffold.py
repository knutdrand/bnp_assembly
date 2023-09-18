from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import pandas as pd
from bionumpy import Genome
from numpy.testing import assert_array_equal
from scipy.stats import poisson

from bnp_assembly.clip_mapper import ClipMapper
from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.distance_distribution import distance_dist
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.expected_edge_counts import ExpectedEdgeCounts, CumulativeDistribution
from bnp_assembly.forbes_score import get_pair_counts, get_node_side_counts, get_forbes_matrix
from bnp_assembly.io import PairedReadStream
from bnp_assembly.missing_data import get_binned_read_counts, find_regions_with_missing_data_from_bincounts, \
    adjust_counts_by_missing_data, find_missing_regions_at_start_and_end_of_contigs
from bnp_assembly.orientation_weighted_counter import OrientationWeightedCounter, OrientationWeightedCountesWithMissing, \
    add_dict_counts, create_distance_matrix
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.location import LocationPair
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.noise_distribution import NoiseDistribution
from bnp_assembly.plotting import px as px_func
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.scaffold_splitting.binned_bayes import NewSplitter
from bnp_assembly.splitting import YahsSplitter, split_on_scores
import logging

from bnp_assembly.datatypes import StreamedGenomicLocationPair

logger = logging.getLogger(__name__)

PathFinder = nxPathFinder


def _split_contig(distance_matrix, path, T=-0.1):
    px_func('debug').histogram([distance_matrix[edge] for edge in path.edges if distance_matrix[edge] > -0.6],
                               nbins=15).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def split_contig_poisson(contig_path, contig_dict, cumulative_distribution, threshold, distance_matrix, n_read_pairs, max_distance=100000):
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


def make_scaffold(genome: Genome,
                  reads: Union[PairedReadStream, Iterable[Union[GenomicLocationPair, StreamedGenomicLocationPair]]],
                  *args,
                  **kwargs) -> Scaffolds:
    contig_sizes, contig_name_translation = get_numeric_contig_name_translation(genome)

    if not isinstance(reads, PairedReadStream):
        # old format, convert to numeric
        reads = (genomic_location_pair.get_numeric_locations() for genomic_location_pair in
                 reads)

    contig_paths = make_scaffold_numeric(contig_sizes, reads, *args, **kwargs)
    scaffold = Scaffolds.from_contig_paths(contig_paths, contig_name_translation)
    return scaffold


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


def get_forbes_counts(read_pairs, contig_dict, cumulative_distribution, bin_size=1000, max_distance=100000):
    forbes_obj = OrientationWeightedCounter(contig_dict,
                                            cumulative_length_distribution=cumulative_distribution,
                                            max_distance=max_distance)
    if isinstance(read_pairs, LocationPair):
        read_pairs = [read_pairs]
    for chunk in read_pairs:
        forbes_obj.register_location_pairs(chunk)
    return forbes_obj.counts


def default_make_scaffold(contig_dict, read_pairs: Iterable[LocationPair], threshold=0.2, max_distance=100000, bin_size=5000):
    distance_matrix = create_distance_matrix_from_reads(contig_dict, read_pairs, max_distance=max_distance)
    path = join_all_contigs(distance_matrix)
    logger.info(f"Joined contigs: {path}")
    s = SplitterInterface(contig_dict, next(read_pairs), path, max_distance=max_distance, bin_size=bin_size, threshold=threshold)
    return s.split()


def get_missing_region_counts(contig_dict, read_pairs, bin_size):
    all_counts = Counter()
    if isinstance(read_pairs, LocationPair):
        read_pairs = [read_pairs]
    for chunk in read_pairs:
        local_counts, bin_sizes = get_binned_read_counts(bin_size, contig_dict, chunk)
        all_counts = add_dict_counts(all_counts, local_counts)
    return all_counts, bin_sizes


def create_distance_matrix_from_reads(contig_dict, read_pairs: Iterable[LocationPair], bin_size=1000, max_distance=100000):
    cumulative_distribution = distance_dist(next(read_pairs), contig_dict)
    bins, bin_sizes = get_missing_region_counts(contig_dict, next(read_pairs), bin_size)
    regions, reads_per_bp = find_regions_with_missing_data_from_bincounts(bin_size, bin_sizes, bins)
    contig_clips = find_missing_regions_at_start_and_end_of_contigs(contig_dict, regions)
    new_contig_dict = {contig_id: end-start for contig_id, (start, end) in contig_clips.items()}
    clip_mapper = ClipMapper(contig_clips)

    mapped_stream = clip_mapper.map_maybe_stream(next(read_pairs))
    counts = get_forbes_counts(mapped_stream, new_contig_dict,
                               cumulative_distribution, bin_size,
                               max_distance=max_distance)
    # adjusted_counts = adjust_counts_by_missing_data(counts, contig_dict, regions, cumulative_distribution, reads_per_bp, max_distance)
    assert np.all(~np.isnan(list(counts.values())))
    # adjusted_counts = adjust_for_missing_data(counts, contig_dict, cumulative_distribution, bin_sizes)
    # forbes_obj = OrientationWeightedCountesWithMissing(contig_dict, next(read_pairs), cumulative_distribution)
    # distance_matrix = forbes_obj.get_distance_matrix()
    distance_matrix = create_distance_matrix(len(contig_dict), counts)
    distance_matrix.plot(name='forbes3')
    return distance_matrix


def make_scaffold_numeric(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', threshold=0.2,
                          bin_size=5000, splitting_method='poisson', max_distance=100000, **distance_kwargs):
    if distance_measure == 'forbes3' and splitting_method != 'poisson':
        return default_make_scaffold(contig_dict, read_pairs, threshold=threshold, max_distance=max_distance, bin_size=bin_size)
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


def join_all_contigs(distance_matrix):
    mapping = None
    for _ in range(len(distance_matrix) // 2):
        paths = PathFinder(distance_matrix).run()
        print([path.directed_nodes for path in paths])
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            return ContigPath.from_node_sides(mapping.popitem()[1])
    assert False
