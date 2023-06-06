from .hic_distance_matrix import calculate_distance_matrices
from .forbes_score import calculate_distance_matrices as forbes_matrix, get_pair_counts, get_node_side_counts, \
    get_pscore_matrix
from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.splitting import LinearSplitter2, LinearSplitter3, YahsSplitter
from bnp_assembly.simulation.hic import simulate_merged_contig_reads, simulate_many_contigs, SimulationParams, \
    full_simulation
from bnp_assembly.location import LocationPair
from bnp_assembly.scaffold import scaffold
from dataclasses import dataclass
import numpy as np


def get_distance_matrix(simulation_params, rng, distance_measure='window', **distance_kwargs):
    n_nodes, n_reads = (simulation_params.n_nodes, simulation_params.n_reads)
    split_and_pairs = simulate_merged_contig_reads(simulation_params.node_length, n_nodes, n_reads,
                                                   p=1 / simulation_params.mean_distance, rng=rng)
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


@dataclass
class SplittingParams:
    bin_size: int = 100
    threshold: float = 0.5


def run_simulated_split_experiment(simulation_params: SimulationParams, rng: object,
                                   splitting_params: SplittingParams = SplittingParams()) -> object:
    #n_nodes, n_reads, n_chromosomes = (
    #    simulation_params.n_nodes, simulation_params.n_reads, simulation_params.n_chromosomes)
    n_chromosomes = simulation_params.n_chromosomes
    split_and_pairs = full_simulation(simulation_params, rng)
    # split_and_pairs = simulate_many_contigs(n_chromosomes, simulation_params.node_length, n_nodes, n_reads, rng=rng,
    #                                        p=1 / simulation_params.mean_distance)
    # split_and_pairs = simulate_merged_contig_reads(simulation_params.node_length, n_nodes, n_reads, rng=rng)
    true_paths = split_and_pairs.split.get_paths()
    assert len(true_paths) == n_chromosomes, true_paths

    #  assert len(split_and_pairs.split.starts) == n_nodes
    contig_dict = split_and_pairs.split.get_contig_dict()
    contig_path = ContigPath.from_directed_nodes(
        DirectedNode(i, '+') for i in split_and_pairs.split.get_contig_dict())

    # split = ScaffoldSplitter3(contig_dict, bin_size).split(contig_path, locations_pair, threshold)
    paths = YahsSplitter(contig_dict, bin_size=splitting_params.bin_size).split(contig_path,
                                                                                LocationPair(split_and_pairs.location_a,
                                                                                             split_and_pairs.location_b),
                                                                                threshold=splitting_params.threshold)
    # splitter = LinearSplitter3(contig_dict, contig_path, window_size=100)
    # npaths = splitter.split(LocationPair(split_and_pairs.location_a,
    #                                    split_and_pairs.location_b))

    return true_paths, paths


def score_scaffolding(true_paths, scaffolded_paths):
    true_edges = {e for path in true_paths for e in path.edges}
    true_edges |= {e.reverse() for path in true_paths for e in path.edges}
    found_edges = {e for path in scaffolded_paths for e in path.edges}
    if len(found_edges) > 0:
        precision = len(true_edges & found_edges) / len(found_edges)
    else:
        precision = 1
    recall = len(true_edges & found_edges) / (len(true_edges) / 2)
    return precision, recall
