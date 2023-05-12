from .hic_distance_matrix import calculate_distance_matrices
from .forbes_score import calculate_distance_matrices as forbes_matrix, get_pair_counts, get_node_side_counts, get_pscore_matrix
from bnp_assembly.simulation.hic import simulate_merged_contig_reads
from bnp_assembly.location import LocationPair
from bnp_assembly.scaffold import scaffold
from dataclasses import dataclass
import numpy as np


def get_distance_matrix(simulation_params, rng, distance_measure='window', **distance_kwargs):
    n_nodes, n_reads = (simulation_params.n_nodes, simulation_params.n_reads)
    split_and_pairs = simulate_merged_contig_reads(simulation_params.node_length, n_nodes, n_reads, p=1/simulation_params.mean_distance, rng=rng)
    assert len(split_and_pairs.split.starts) == n_nodes
    contig_dict = split_and_pairs.split.get_contig_dict()
    read_pairs = LocationPair(split_and_pairs.location_a,
                              split_and_pairs.location_b)
    if distance_measure == 'window':
        return calculate_distance_matrices(contig_dict, read_pairs)
    elif distance_measure == 'forbes':
        p = get_pair_counts(contig_dict, read_pairs)
        n = get_node_side_counts(p)
        get_pscore_matrix(p, n).plot().show()
        return forbes_matrix(contig_dict, read_pairs, **distance_kwargs)


def run_simulated_experiment(simulation_params, rng, distance_measure='window'):
    n_nodes, n_reads = (simulation_params.n_nodes, simulation_params.n_reads)
    split_and_pairs = simulate_merged_contig_reads(simulation_params.node_length, n_nodes, n_reads, rng=rng)
    assert len(split_and_pairs.split.starts) == n_nodes
    paths = scaffold(split_and_pairs.split.get_contig_dict(),
                     LocationPair(split_and_pairs.location_a,
                                  split_and_pairs.location_b),
                     distance_measure=distance_measure,
                     window_size=30)
    true_paths = split_and_pairs.split.get_paths()
    return true_paths, paths



def score_scaffolding(true_paths, scaffolded_paths):
    true_edges = {e for path in true_paths for e in path.edges}
    true_edges |= {e.reverse() for path in true_paths for e in path.edges}
    found_edges = {e for path in scaffolded_paths for e in path.edges}
    if len(found_edges)>0:
        precision = len(true_edges & found_edges)/len(found_edges)
    else:
        precision = 1
    recall = len(true_edges & found_edges)/(len(true_edges)/2)
    return precision, recall
