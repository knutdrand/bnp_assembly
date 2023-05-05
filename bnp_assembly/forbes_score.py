import numpy as np
from .location import LocationPair, Location
from .graph_objects import NodeSide, Edge
from .hic_distance_matrix import DirectedDistanceMatrix
from collections import Counter
import typing as tp

def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair):
    node_side_counts, pair_counts = count_window_combinastions(contig_dict, location_pairs)
    distance_matrix = DirectedDistanceMatrix(len(contig_dict))
    N = sum(node_side_counts.values())
    for contig_a in contig_dict:
        for contig_b in contig_dict:
            for dir_a, dir_b in (('r', 'l'), ('r', 'r'), ('l', 'l'), ('l', 'r')):
                node_side_a = NodeSide(contig_a, dir_a)
                node_side_b = NodeSide(contig_b, dir_b)
                edge = Edge(node_side_a, node_side_b)
                score = (pair_counts[edge]+1/len(contig_dict))/((node_side_counts[node_side_a]+1)*(node_side_counts[node_side_b])+1)
                distance_matrix[edge] = score
                distance_matrix[Edge(node_side_b, node_side_a)] = score
    return distance_matrix
                

def _naive_side_weight(position, size):
    return position/size

def count_window_combinastions(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, side_weight_func=_naive_side_weight) -> tp.Tuple[Counter, Counter]:
    node_side_counts = Counter()
    pair_counts = Counter()
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        if a.contig_id == b.contig_id:
            continue
        right_weight_a = side_weight_func(a.offset, contig_dict[int(a.contig_id)])
        right_weight_b = side_weight_func(b.offset, contig_dict[int(b.contig_id)])
        for direction_a in ('l', 'r'):
            for direction_b in ('l', 'r'):
                a_side = NodeSide(int(a.contig_id), direction_a)
                b_side = NodeSide(int(b.contig_id), direction_b)
                p_a = right_weight_a if direction_a == 'r' else 1-right_weight_a
                p_b = right_weight_b if direction_b == 'r' else 1-right_weight_b
                node_side_counts[a_side] += p_a
                node_side_counts[b_side] += p_b
                pair_counts[Edge(a_side, b_side)] += p_a*p_b
                pair_counts[Edge(b_side, a_side)] += p_a*p_b                

    return node_side_counts, pair_counts
    
