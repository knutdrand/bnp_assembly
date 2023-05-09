from dataclasses import dataclass
import typing as tp
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import poisson
from bnp_assembly.location import LocationPair, Location
from .contig_graph import ContigGraph
from .graph_objects import NodeSide, Edge
from .distance_matrix import DirectedDistanceMatrix
import logging
logger = logging.getLogger(__name__)


def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, window_size=15):
    logger.info('Calculating distance matrix')
    overlap_counts, inside_counts = count_window_combinastions(contig_dict, location_pairs, window_size)
    all_edges = defaultdict(lambda: defaultdict(dict))
    distance_matrix = DirectedDistanceMatrix(len(contig_dict))
    for contig_a in contig_dict:
        for contig_b in contig_dict:
            for dir_a, dir_b in (('r', 'l'), ('r', 'r'), ('l', 'l'), ('l', 'r')):
                node_side_a = NodeSide(contig_a, dir_a)
                node_side_b = NodeSide(contig_b, dir_b)
                edge = Edge(node_side_a, node_side_b)
                if contig_a == contig_b:
                    distance_matrix[edge] = np.inf
                    continue

                id_a = (contig_a, dir_a)
                id_b = (contig_b, dir_b)
                overlap_count = overlap_counts[frozenset([id_a, id_b])]
                score = calc_score(inside_counts[id_a],
                                   inside_counts[id_b],
                                   overlap_count)
                distance_matrix[edge] = score

    return distance_matrix



def count_window_combinastions(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, window_size=15) -> Counter:
    overlap_counts = Counter()
    inside_counts = Counter()
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        for direction_a in ('l', 'r'):
            for direction_b in ('l', 'r'):
                if direction_a == 'l':
                    offset_a = a.offset
                else:
                    offset_a = contig_dict[int(a.contig_id)]-a.offset
                if direction_b == 'l':
                    offset_b = b.offset
                else:
                    offset_b = contig_dict[int(b.contig_id)]-b.offset
                id_a = (int(a.contig_id), direction_a)
                id_b = (int(b.contig_id), direction_b)
                if (a.contig_id != b.contig_id) and (offset_a < window_size) and (offset_b<window_size):
                    overlap_counts[frozenset((id_a, id_b))] += 1
                
                elif id_a == id_b:
                    if min(offset_a, offset_b) < window_size and window_size <= max(offset_a, offset_b) < 2*window_size:
                        inside_counts[id_a] += 1
    return overlap_counts, inside_counts


def calc_score(inside_a_count, inside_b_count, overlap_count):
    rate = np.mean([inside_a_count, inside_b_count])
    return -(overlap_count+0.1)/(rate+1)
