from dataclasses import dataclass

import numpy as np
import bionumpy as bnp
import pytest
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices, count_window_combinastions, NodeSide, Edge
from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.path_finding import best_path
from bnp_assembly.simulation.hic_read_simulation import simulate_hic_from_file, simulate_raw, \
    PairedReadPositionsDistribution, simulate

np.random.seed(42)


@pytest.fixture
def contig_list():
    return {0: 100, 1: 50}


@pytest.fixture
def location_pairs():
    location_a = Location.from_entry_tuples([
        (0, 10),
        (1, 40),
        (0, 8)])
    location_b = Location.from_entry_tuples([
        (1, 45),
        (0, 5),
        (0, 20)])
    return LocationPair(location_a, location_b)


@pytest.fixture
def contig_list2():
    return {0: 10, 1: 10, 2: 10}


@pytest.fixture
def location_pairs2():
    location_a = Location.from_entry_tuples([
        (1, 0),
        (2, 1)])
    location_b = Location.from_entry_tuples([
        (2, 9),
        (0, 0)])

    return LocationPair(location_a, location_b)


def test_count_window_combinations(contig_list, location_pairs):
    overlap_count, inside_counts = count_window_combinastions(contig_list, location_pairs)
    # print(overlap_count)
    assert overlap_count[frozenset(((0, 'l'), (1, 'r')))] == 2
    assert sum(overlap_count.values()) == 2
    assert inside_counts[(0, 'l')] == 1


def test_distance_best_path(contig_list, location_pairs):
    graph = calculate_distance_matrices(contig_list, location_pairs)
    path = best_path(graph)[0]
    a = path.to_list()
    # print(a)
    b = path.reverse().to_list()
    correct_path = [(1, 0), (0, 0)]
    assert (a == correct_path or b == correct_path)


def test_count_window_combinations(contig_list2, location_pairs2):
    overlap_count, inside_counts = count_window_combinastions(contig_list2, location_pairs2, window_size=5)
    # print(overlap_count)
    assert overlap_count[frozenset(((1, 'l'), (2, 'r')))] == 1
    assert overlap_count[frozenset(((2, 'l'), (0, 'l')))] == 1
    assert sum(overlap_count.values()) == 2
    assert sum(inside_counts.values()) == 0

def test_distance_distance_matrtix(contig_list2, location_pairs2):
    graph = calculate_distance_matrices(contig_list2, location_pairs2, window_size=5)
    data = graph.data
    assert np.argmin(data[2]) == 5, data


def test_distance_best_path2(contig_list2, location_pairs2):
    graph = calculate_distance_matrices(contig_list2, location_pairs2, window_size=5)
    path = best_path(graph)[0]
    a = path.to_list()
    b = path.reverse().to_list()
    correct_path = [(1, 1), (2, 1), (0, 0)]
    assert (a == correct_path or b == correct_path)


def get_node_side_location(node_side: NodeSide, node_length=100):
    if node_side.side == 'l':
        return (node_side.node_id, 1)
    else:
        return (node_side.node_id, node_length-2)


def generate_reads_for_path(edges):
    from_locations = Location.from_entry_tuples([get_node_side_location(edge.from_node_side) for edge in edges])
    to_locations = Location.from_entry_tuples([get_node_side_location(edge.to_node_side) for edge in edges])
    return LocationPair(from_locations, to_locations)

def generate_random_edges(n_nodes):
    ordering = np.arange(n_nodes)
    np.random.shuffle(ordering)
    reverse = np.random.choice([0, 1], size=n_nodes)
    edges = []
    for i in range(n_nodes-1):
        from_node = NodeSide(ordering[i], 'r' if reverse[i] == 0 else 'l')
        to_node = NodeSide(ordering[i+1], 'l' if reverse[i+1] == 0 else 'r')
        edges.append(Edge(from_node, to_node))
    return edges


def generate_random_order(n_nodes, signal=2, noise=1):
    contig_dict = {i: 100 for i in range(n_nodes)}
    edges = generate_random_edges(n_nodes)
    correct_path = ContigPath.from_edges(edges)
    pairs = generate_reads_for_path(edges)
    false_pairs = generate_reads_for_path(generate_random_edges(n_nodes))
    all_pairs = LocationPair(np.concatenate([pairs.location_a]*signal+[false_pairs.location_a]*noise),
                             np.concatenate([pairs.location_b]*signal+[false_pairs.location_b]*noise))
    return contig_dict, correct_path, all_pairs


@pytest.mark.parametrize('n_nodes', list(range(3, 20)))
@pytest.mark.parametrize('s', list(range(2, 10)))
def test_random(n_nodes, s):
    contig_dict, correct_path, pairs = generate_random_order(n_nodes, signal=s, noise=s-1)
    print(correct_path, pairs)
    graph = calculate_distance_matrices(contig_dict, pairs)
    assert best_path(graph) == [correct_path]

@dataclass
class Dummy:
    sequence: str


def test_hic_simulation_acceptance():
    contigs = bnp.as_encoded_array(['A'*1000, 'B'*500, 'C'*700])
    contigs= Dummy(contigs)
    data = PairedReadPositionsDistribution(contigs, 200,  20, 0.5).sample(100)
    assert len(data) == 100
    # simulate_raw(contigs, 200, 100, 20, 0.5)


def test_simulate():
    contigs = bnp.as_encoded_array(['A'*1000, 'B'*500, 'C'*700])
    contigs= Dummy(contigs)
    data_stream = simulate(contigs, 100, 20, 200, 0.5)
    # data = PairedReadPositionsDistribution(contigs, 200,  20, 0.5).sample(100)
    stream = [len(data) == 100 for data in data_stream]
    assert all(stream)
