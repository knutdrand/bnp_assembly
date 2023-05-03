from itertools import product
from more_itertools import chunked
import typing as tp
from collections import defaultdict
from .contig_graph import ContigGraph, ContigPath
from .hic_distance_matrix import NodeSide, Edge, DirectedDistanceMatrix


def create_merged_graph(paths: tp.List[ContigPath], distance_matrix, path_mapping: tp.Dict[int, tp.List[NodeSide]]=None):
    paths = [path.node_sides for path in paths]
    if path_mapping is None:
        path_mapping = {node_id: [NodeSide(node_id, 'l'), NodeSide(node_id, 'r')]
                        for node_id in range(len(distance_matrix)//2)}
    new_distance_matrix = distance_matrix.__class__(len(paths))

    side_map = {'l': 0, 'r': -1}
    for (i, path_1), (j, path_2) in product(enumerate(paths), repeat=2):
        for side_1, side_2 in product(('l', 'r'), repeat=2):
            old_edge = Edge(path_1[side_map[side_1]], path_2[side_map[side_2]])
            score = distance_matrix[old_edge]
            new_distance_matrix[Edge(NodeSide(i, side_1), NodeSide(j, side_2))] = score

    new_mapping = defaultdict(list)
    for new_node_id, path in enumerate(paths):
        for node_side_1, _ in chunked(path, 2):
            if node_side_1.side == 'l':
                new_mapping[new_node_id].extend(path_mapping[node_side_1.node_id])
            else:
                new_mapping[new_node_id].extend(path_mapping[node_side_1.node_id][::-1])

    return new_distance_matrix, new_mapping
        

    

# def iterative_join(path_finder_class, distance_matrix):
#     paths = path_finder_class(distance_matrix).run()
    
