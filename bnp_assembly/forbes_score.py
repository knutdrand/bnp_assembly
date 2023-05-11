import numpy as np
from .location import LocationPair, Location
from .graph_objects import NodeSide, Edge
from .distance_matrix import DirectedDistanceMatrix
from .distance_distribution import calculate_distance_distritbution
from collections import Counter, defaultdict
from scipy.stats import poisson
import plotly.express as px
import typing as tp


def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs):
    F = calculate_distance_distritbution(list(contig_dict.values()),
                                         [np.abs(a.offset-b.offset)
                                          for a, b in zip(location_pairs.location_a, location_pairs.location_b)
                                          if a.contig_id == b.contig_id])

    node_side_counts, pair_counts = count_window_combinastions(contig_dict, location_pairs, CumulativeSideWeight(F))
    return  get_forbes_matrix(pair_counts, node_side_counts, contig_dict)
    
    distance_matrix = DirectedDistanceMatrix(len(contig_dict))

    N = sum(node_side_counts.values())
    alpha = 1
    for contig_a in contig_dict:
        for contig_b in contig_dict:
            for dir_a, dir_b in (('r', 'l'), ('r', 'r'), ('l', 'l'), ('l', 'r')):
                node_side_a = NodeSide(contig_a, dir_a)
                node_side_b = NodeSide(contig_b, dir_b)
                edge = Edge(node_side_a, node_side_b)
                score = N*(pair_counts[edge]+alpha/(len(contig_dict)*4))/((node_side_counts[node_side_a]+alpha)*(node_side_counts[node_side_b])+alpha)
                distance_matrix[edge] = -np.log(score)
                distance_matrix[Edge(node_side_b, node_side_a)] = -np.log(score)
    return distance_matrix

def get_pvalue_matrix(pair_counts: tp.Dict[Edge, float], node_side_counts: tp.Dict[NodeSide, float]):
    n_nodes = len(node_side_counts)//2
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = sum(node_side_counts.values())/2
    for edge, value in pair_counts.items():
        rate = node_side_counts[edge.from_node_side]*node_side_counts[edge.to_node_side]/N
        print(edge,rate, N)
        p_value = poisson.sf(pair_counts[edge], rate)
        distance_matrix[edge] = p_value
        distance_matrix[edge.reverse()] = p_value
    return distance_matrix


def get_forbes_matrix(pair_counts, node_side_counts, contig_dict, alpha=1):
    n_nodes = len(node_side_counts)//2
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = sum(node_side_counts.values())/2
    # alpha = 1
    for contig_a in contig_dict:
        for contig_b in contig_dict:
            for dir_a, dir_b in (('r', 'l'), ('r', 'r'), ('l', 'l'), ('l', 'r')):
                node_side_a = NodeSide(contig_a, dir_a)
                node_side_b = NodeSide(contig_b, dir_b)
                edge = Edge(node_side_a, node_side_b)
                score = N*(pair_counts[edge]+alpha/(n_nodes*4))/((node_side_counts[node_side_a]+alpha)*(node_side_counts[node_side_b])+alpha)
                distance_matrix[edge] = -np.log(score)
                distance_matrix[Edge(node_side_b, node_side_a)] = -np.log(score)
    return distance_matrix
    
                

def _naive_side_weight(position, size):
    return position/size


class CumulativeSideWeight:
    def __init__(self, cumulative_distribution):
        self.F = cumulative_distribution

    def __call__(self, position, size):
        '''
        P(s='R'| other=outside) = P(other=outside|s='r')P(s='r')/P(other=outside)
        P(s='r') = 0.5
        P(other=outside|s='r') = 1-F(size-position-1)
        P(other=outside|s='l') = 1-F(position)
        '''
        wR = (1-self.F[size-position-1])
        wL = (1-self.F[position])
        return wR/(wR+wL)


def count_window_combinastions(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, side_weight_func=_naive_side_weight) -> tp.Tuple[Counter, Counter]:
    node_side_counts = defaultdict(float)
    pair_counts = defaultdict(float)
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        if a.contig_id == b.contig_id:
            continue
        right_weight_a = side_weight_func(a.offset, contig_dict[int(a.contig_id)])
        right_weight_b = side_weight_func(b.offset, contig_dict[int(b.contig_id)])
        for direction_a in ('l', 'r'):
            a_side = NodeSide(int(a.contig_id), direction_a)
            p_a = right_weight_a if direction_a == 'r' else 1-right_weight_a
            # node_side_counts[a_side] += p_a
            for direction_b in ('l', 'r'):
                b_side = NodeSide(int(b.contig_id), direction_b)
                p_b = right_weight_b if direction_b == 'r' else 1-right_weight_b
                # node_side_counts[b_side] += p_b
                pair_counts[Edge(a_side, b_side)] += p_a*p_b
                pair_counts[Edge(b_side, a_side)] += p_a*p_b
            # n = node_side_counts[a_side]
            # p = sum(value for key, value in pair_counts.items() if key.from_node_side == a_side)
            # assert n==p, (n, p, p_a, p_b)
    # for contig_id in contig_dict:
    #     for direction in ('l', 'r'):
    #         node_side = NodeSide(contig_id, direction)
    #         n = node_side_counts[node_side]
    #         p = sum(value for key, value in pair_counts.items() if key.from_node_side == node_side)            
    #         assert n == p, (n, p)
    node_side_counts = {node_side: sum(value for key, value in pair_counts.items() if key.from_node_side == node_side) for node_side in (NodeSide(contig_id, direction) for contig_id in contig_dict for direction in ('l', 'r'))}
    return node_side_counts, pair_counts
    
