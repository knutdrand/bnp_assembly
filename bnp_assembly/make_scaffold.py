import pandas as pd
from numpy.testing import assert_array_equal

from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.forbes_score import get_pair_counts, get_node_side_counts, get_forbes_matrix
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices
from bnp_assembly.iterative_join import create_merged_graph
from bnp_assembly.location import LocationPair
from bnp_assembly.networkx_wrapper import PathFinder as nxPathFinder
from bnp_assembly.plotting import px as px_func
from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
from bnp_assembly.splitting import YahsSplitter

PathFinder = nxPathFinder


def _split_contig(distance_matrix, path, T=-0.1):
    px('debug').histogram([distance_matrix[edge] for edge in path.edges if distance_matrix[edge]>-0.6], nbins=15).show()
    split_edges = (edge for edge in path.edges if distance_matrix[edge] >= T)
    return path.split_on_edges(split_edges)


def split_contig(contig_path, contig_dict, threshold, bin_size, locations_pair):
    # return LinearSplitter3(contig_dict,  contig_path).split(locations_pair)
    #return LinearSplitter2(contig_dict,  contig_path).split(locations_pair)
    YahsSplitter.matrix_class = BinnedBayes
    return YahsSplitter(contig_dict, bin_size).split(contig_path, locations_pair)

    #return ScaffoldSplitter3(contig_dict, bin_size).split(contig_path, locations_pair, threshold)

    # return LinearSplitter(contig_dict, threshold).iterative_split(contig_path, locations_pair)


def make_scaffold(contig_dict: dict, read_pairs: LocationPair, distance_measure='window', threshold=0.0, bin_size=5000, **distance_kwargs):
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
        # split_matrix = calculate_distance_matrices(contig_dict, read_pairs, **distance_kwargs)
        # split_matrix = get_pscore_matrix(pair_counts, node_side_counts)
        original_distance_matrix.plot(name='forbes')

    distance_matrix = original_distance_matrix
    assert_array_equal(distance_matrix.data.T, distance_matrix.data)
    mapping = None
    for _ in range(len(distance_matrix)//2):
        paths = PathFinder(distance_matrix).run()
        distance_matrix, mapping = create_merged_graph(paths, distance_matrix, mapping)
        if len(mapping) == 1:
            path = ContigPath.from_node_sides(mapping.popitem()[1])
            paths = split_contig(path, contig_dict, -threshold, bin_size, read_pairs)
            return paths
    assert len(mapping) == 0, mapping
