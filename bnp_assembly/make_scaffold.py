import pandas as pd
from bionumpy import Genome
from numpy.testing import assert_array_equal

from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.forbes_score import get_pair_counts, get_node_side_counts, get_forbes_matrix, Forbes2, \
    ForbesWithMissingData
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.location import LocationPair
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.plotting import px as px_func
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
from bnp_assembly.splitting import YahsSplitter

PathFinder = nxPathFinder


def _split_contig(distance_matrix, path, T=-0.1):
    px_func('debug').histogram([distance_matrix[edge] for edge in path.edges if distance_matrix[edge]>-0.6], nbins=15).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


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


def make_scaffold_numeric(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', threshold=0.0, bin_size=5000, **distance_kwargs):
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
        px.bar(x=[str(ns) for ns in node_side_counts.keys()], y=list(node_side_counts.values()), title='node side counts')
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
    for _ in range(len(distance_matrix)//2):
        paths = PathFinder(distance_matrix).run()
        print([path.directed_nodes for path in paths])
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            path = ContigPath.from_node_sides(mapping.popitem()[1])
            paths = split_contig(path, contig_dict, -threshold, bin_size, read_pairs)
            return paths
    assert len(mapping) == 0, mapping
