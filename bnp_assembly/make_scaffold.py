import numpy as np
import pandas as pd
from bionumpy import Genome
from numpy.testing import assert_array_equal
from scipy.stats import poisson

from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.distance_distribution import distance_dist
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.expected_edge_counts import ExpectedEdgeCounts, CumulativeDistribution
from bnp_assembly.forbes_score import get_pair_counts, get_node_side_counts, get_forbes_matrix, Forbes2, \
    ForbesWithMissingData
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.location import LocationPair
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.plotting import px as px_func
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
from bnp_assembly.splitting import YahsSplitter, split_on_scores

PathFinder = nxPathFinder


def _split_contig(distance_matrix, path, T=-0.1):
    px_func('debug').histogram([distance_matrix[edge] for edge in path.edges if distance_matrix[edge] > -0.6],
                               nbins=15).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def split_contig_poisson(contig_path, contig_dict, cumulative_distribution, threshold, distance_matrix, n_read_pairs):
    expected_edge_counts = ExpectedEdgeCounts(contig_dict, cumulative_distribution)
    p_value_func = lambda observed, expected: poisson.cdf(observed, expected)
    observed = {edge: np.exp(-distance_matrix[edge]) for edge in contig_path.edges}
    expected = {edge: expected_edge_counts.get_expected_edge_count(edge) for edge in contig_path.edges}

    # ratio = sum(observed.values())/sum(expected.values())
    expected= {edge: expected[edge]*n_read_pairs for edge in contig_path.edges}
    table = [(str(edge), contig_dict[edge.from_node_side.node_id], contig_dict[edge.to_node_side.node_id],
              observed[edge], expected[edge])
             for edge in contig_path.edges]
    frame = pd.DataFrame(table, columns=['name', 'length_a', 'length_b', 'observed', 'expected'])
    px_func(name='splitting').scatter(data_frame=frame, x='length_a', y='length_b', color='observed', size='expected',
                                      title='sizedep')
    px_func(name='splitting').scatter(data_frame=frame, x='observed', y='expected', title='observed vs expected')

    edge_scores = {
        edge: p_value_func(observed[edge], expected[edge])
        for edge in contig_path.edges}
    return split_on_scores(contig_path, edge_scores, threshold, keep_over=True)


def split_contig(contig_path, contig_dict, threshold, bin_size, locations_pair):
    YahsSplitter.matrix_class = BinnedBayes
    return YahsSplitter(contig_dict, bin_size).split(contig_path, locations_pair, threshold=threshold)


def make_scaffold(genome: Genome, genomic_location_pair: GenomicLocationPair, *args, **kwargs) -> Scaffolds:
    encoding = genome.get_genome_context().encoding
    contig_dict = genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    numeric_contig_dict = {int(encoding.encode(name).raw()): value for name, value in contig_dict.items()}
    numeric_locations_pair = genomic_location_pair.get_numeric_locations()
    contig_paths = make_scaffold_numeric(numeric_contig_dict, numeric_locations_pair, *args, **kwargs)
    scaffold = Scaffolds.from_contig_paths(contig_paths, translation_dict)
    return scaffold


def make_scaffold_numeric(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', threshold=0.0,
                          bin_size=5000, splitting_method='poisson', **distance_kwargs):
    px = px_func(name='joining')
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
        original_distance_matrix = Forbes2(contig_dict, read_pairs).get_distance_matrix()
        original_distance_matrix.plot(name='forbes2')
    elif distance_measure == 'forbes3':
        original_distance_matrix = ForbesWithMissingData(contig_dict, read_pairs).get_distance_matrix()
        original_distance_matrix.plot(name='forbes3')
    original_distance_matrix.inversion_plot('forbes2')
    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    mapping = None
    for _ in range(len(distance_matrix) // 2):
        paths = PathFinder(distance_matrix).run()
        print([path.directed_nodes for path in paths])
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            path = ContigPath.from_node_sides(mapping.popitem()[1])
            break
    if splitting_method == 'poisson':
        cumulative_distribution = CumulativeDistribution(
            distance_dist(read_pairs, contig_dict),
            p_noise=0.01,
            genome_size=sum(contig_dict.values()))
        paths = split_contig_poisson(path, contig_dict, cumulative_distribution, threshold, original_distance_matrix, len(read_pairs.location_b))
    else:
        paths = split_contig(path, contig_dict, -threshold, bin_size, read_pairs)
    return paths