import numpy as np
from .location import LocationPair, Location
from .graph_objects import NodeSide, Edge
from .distance_matrix import DirectedDistanceMatrix
from .distance_distribution import calculate_distance_distritbution, distance_dist
from collections import Counter, defaultdict
from scipy.stats import poisson
from .plotting import px
import typing as tp

def get_pair_counts(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs): 
    F = distance_dist(location_pairs, contig_dict)
    px("debug").line(F).show()
    # calculate_distance_distritbution(list(contig_dict.values()),
    #[np.abs(a.offset-b.offset)
    #                                      for a, b in zip(location_pairs.location_a, location_pairs.location_b)
    #                                      if a.contig_id == b.contig_id])
    return count_window_combinastions(contig_dict, location_pairs, CumulativeSideWeight(F))

def get_node_side_counts(pair_counts):
    node_sides = {edge.from_node_side for edge in pair_counts} | {edge.to_node_side for edge in pair_counts}
    return {node_side: sum((pair_counts[Edge(node_side, node_side2)]+pair_counts[Edge(node_side2, node_side)])/2 for node_side2 in node_sides if True or node_side2.node_id != node_side.node_id) for node_side in node_sides}
#     return {node_side: sum(value for key, value in pair_counts.items() if key.from_node_side == node_side) for node_side in node_sides}

def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, **kwargs):
    pair_counts = get_pair_counts(contig_dict, location_pairs)
    node_side_counts = get_node_side_counts(pair_counts)
    return  get_forbes_matrix(pair_counts, node_side_counts)

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
    for edge, count in pair_counts.items():
        rate = get_rate(node_side_counts[edge.from_node_side], node_side_counts[edge.to_node_side], N, alpha)
        score = (pair_counts[edge]+(alpha/(n_nodes*2)))/rate
        # score = N*(pair_counts[edge]+alpha/(n_nodes*4))/((node_side_counts[edge.from_node_side]+alpha)*(node_side_counts[edge.to_node_side])+alpha)
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
    pair_counts = {Edge(a, b): 0 for a in node_sides for b in node_sides} # if not a.node_id==b.node_id}
    tmp = []
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        #if a.contig_id == b.contig_id:
        #    continue
        right_weight_a = side_weight_func(a.offset, contig_dict[int(a.contig_id)])
        right_weight_b = side_weight_func(b.offset, contig_dict[int(b.contig_id)])
        for direction_a in ('l', 'r'):
            a_side = NodeSide(int(a.contig_id), direction_a)
            p_a = right_weight_a if direction_a == 'r' else 1-right_weight_a
            for direction_b in ('l', 'r'):
                b_side = NodeSide(int(b.contig_id), direction_b)
                p_b = right_weight_b if direction_b == 'r' else 1-right_weight_b
                pair_counts[Edge(a_side, b_side)] += p_a*p_b
                pair_counts[Edge(b_side, a_side)] += p_a*p_b
    return pair_counts
    
