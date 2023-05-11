import numpy as np
from .location import LocationPair, Location
from .graph_objects import NodeSide, Edge
from .distance_matrix import DirectedDistanceMatrix
from .distance_distribution import calculate_distance_distritbution
from collections import Counter, defaultdict
from scipy.stats import poisson
import plotly.express as px
import typing as tp

def get_pair_counts(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs): 
    F = calculate_distance_distritbution(list(contig_dict.values()),
                                         [np.abs(a.offset-b.offset)
                                          for a, b in zip(location_pairs.location_a, location_pairs.location_b)
                                          if a.contig_id == b.contig_id])
    px.line(F).show()
    return count_window_combinastions(contig_dict, location_pairs, CumulativeSideWeight(F))

def get_node_side_counts(pair_counts):
    node_sides = {edge.from_node_side for edge in pair_counts} | {edge.to_node_side for edge in pair_counts}
    return {node_side: sum(value for key, value in pair_counts.items() if key.from_node_side == node_side) for node_side in node_sides}

def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs):
    # F = calculate_distance_distritbution(list(contig_dict.values()),
    #                                      [np.abs(a.offset-b.offset)
    #                                       for a, b in zip(location_pairs.location_a, location_pairs.location_b)
    #                                       if a.contig_id == b.contig_id])
    pair_counts = get_pair_counts(contig_dict, location_pairs)
    print(pair_counts)
    print(node_side_counts)
    # count_window_combinastions(contig_dict, location_pairs, CumulativeSideWeight(F))
    node_side_counts = get_node_side_counts(pair_counts)
    return  get_forbes_matrix(pair_counts, node_side_counts)
    
    distance_matrix = DirectedDistanceMatrix(len(contig_dict))

    N = sum(node_side_counts.values())
    alpha = 1
    for contig_a in contig_dict:
        for contig_b in contig_dict:
            for dir_a, dir_b in (('r', 'l'), ('r', 'r'), ('l', 'l'), ('l', 'r')):
                node_side_a = NodeSide(contig_a, dir_a)
                node_side_b = NodeSide(contig_b, dir_b)
                edge = Edge(node_side_a, node_side_b)
                rate = get_rate(node_side_counts[node_side_a], node_side_counts[node_side_b], N, alpha)
                score = pair_counts[edge]+alpha/(len(contig_dict)*4)
                # score = N*(pair_counts[edge]+alpha/(len(contig_dict)*4))/((node_side_counts[node_side_a]+alpha)*(node_side_counts[node_side_b])+alpha)
                distance_matrix[edge] = -np.log(score)
                distance_matrix[Edge(node_side_b, node_side_a)] = -np.log(score)
    return distance_matrix

def get_rate(count_a, count_b, N, alpha=1):
    return (count_a+alpha)*(count_b+alpha)/N

def get_pscore_matrix(pair_counts: tp.Dict[Edge, float], node_side_counts: tp.Dict[NodeSide, float], alpha=1):
    n_nodes = len(node_side_counts)//2
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = sum(node_side_counts.values()) # Should this be divided by two?
    for edge, value in pair_counts.items():
        rate = get_rate(node_side_counts[edge.from_node_side], node_side_counts[edge.to_node_side], N, 0)
        #node_side_counts[edge.from_node_side]*node_side_counts[edge.to_node_side]/N
        p_score = poisson.sf(pair_counts[edge]-1, rate)
        distance_matrix[edge] = p_score
        distance_matrix[edge.reverse()] = p_score
    return distance_matrix


def get_forbes_matrix(pair_counts, node_side_counts, alpha=1):
    n_nodes = len(node_side_counts)//2
    distance_matrix = DirectedDistanceMatrix(n_nodes)
    N = sum(node_side_counts.values())
    # alpha = 1
    for edge, count in pair_counts.items():
        score = N*(pair_counts[edge]+alpha/(n_nodes*4))/((node_side_counts[edge.from_node_side]+alpha)*(node_side_counts[edge.to_node_side])+alpha)
        distance_matrix[edge] = -np.log(score)
        distance_matrix[edge.reverse()] = -np.log(score)
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
    node_sides = [NodeSide(i, d) for i in range(len(contig_dict)) for d in ('l', 'r')]
    pair_counts = {Edge(a, b): 0 for a in node_sides for b in node_sides}
    tmp = []
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        if a.contig_id == b.contig_id:
            continue
        right_weight_a = side_weight_func(a.offset, contig_dict[int(a.contig_id)])
        right_weight_b = side_weight_func(b.offset, contig_dict[int(b.contig_id)])
        if a.contig_id == 10:
            tmp.append(a.offset)
        if b.contig_id == 10:
            tmp.append(b.offset)
        for direction_a in ('l', 'r'):
            a_side = NodeSide(int(a.contig_id), direction_a)
            p_a = right_weight_a if direction_a == 'r' else 1-right_weight_a
            for direction_b in ('l', 'r'):
                b_side = NodeSide(int(b.contig_id), direction_b)
                p_b = right_weight_b if direction_b == 'r' else 1-right_weight_b
                pair_counts[Edge(a_side, b_side)] += p_a*p_b
                pair_counts[Edge(b_side, a_side)] += p_a*p_b
    # node_side_counts = {node_side: sum(value for key, value in pair_counts.items() if key.from_node_side == node_side) for node_side in (NodeSide(contig_id, direction) for contig_id in contig_dict for direction in ('l', 'r'))}
    px.histogram(tmp).show()
    return pair_counts
    
