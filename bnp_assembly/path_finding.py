import typing as tp
from scipy.optimize import linear_sum_assignment
from .hic_distance_matrix import Edge, DirectedDistanceMatrix, NodeSide
from .contig_graph import ContigPath
from collections import Counter
import numpy as np

class PathFinder:
    def __init__(self, distance_matrix):
        self._distance_matrix = distance_matrix
        np.fill_diagonal(self._distance_matrix.data, np.inf)

    def find_one_contig(self, side_dict):
        first_side, cur_side = side_dict.popitem()
        node_sides = []
        while True:
            node_sides.append(cur_side)
            cur_side = cur_side.other_side()
            node_sides.append(cur_side)

            if cur_side == first_side:
                break
            next_side = side_dict.pop(cur_side)
            cur_side = next_side
        return node_sides


    def _split_path(self, path):
        if any(v>1 for v in Counter(path).values()):
            path = self.find_sub_cycle(path)
        edges = [Edge(*pair) for pair in zip(path[1::2], path[2::2])] + [Edge(path[-1], path[0])]
        scores = [self._distance_matrix[edge] for edge in edges]
        i = np.argmax(scores)
        cut_idx = 2*(i+1)
        path =  path[cut_idx:] + path[:cut_idx]
        return path
        seen = set()
        split_points = [0]
        for i, node_side in enumerate(path):
            if node_side in seen:
                split_points.append(i)
                seen = set()
            seen.add(node_side)
        split_points.append(len(path))
        sub_paths = (path[s:e] for s,e in zip(split_points[:-1], split_points[1:]))
        def path_score(path):
            edges = [Edge(*pair) for pair in zip(path[1::2], path[2::2])]
            return sum(self._distance_matrix[edge] for edge in edges)
        return min(sub_paths, key=path_score)

    def find_sub_cycle(self, path):
        joined_path = path+path
        max_path= []
        n_unique_node_sides = len(set(path))
        max_l  = 0 
        seen = set()
        final_path = []
        for node_side in path:
            if node_side in seen:
                continue
            final_path.append(node_side)
            seen.add(node_side)
        return final_path
# 
#         for i in range(len(path)):
#             cur_seen = set()
#             for j in range(n_unique_node_sides):
#                 cur_idx = (i+j)%len(path)
#                 max_l = max(max_l, j)
#                 if path[cur_idx] in cur_seen:
#                     break
#                 cur_seen.add(path[cur_idx])
#             else:
#                 return [path[(i+j) % len(path)] for j in range(n_unique_node_sides)]
#         assert False, (path, max_l, n_unique_node_sides)
#         
# 
#         start = None
#         for i in range(0, len(path)-2, 2):
#             if path[i-1] == path[i+2]:
#                 if start is not None:
#                     return path[start:i]
#                 start = i
#         assert start is not None, path
#         return path[start:len(path)-2]

    def _split_path_on_seen_nodes(self, path, seen_nodes):
        sub_paths = []
        cur_path = []
        for node_side in path:
            if node_side.node_id in seen_nodes:
                if len(cur_path) > 0:
                    sub_paths.append(cur_path)
                cur_path = []
            else:
                cur_path.append(node_side)
        if len(cur_path) > 0:
            sub_paths.append(cur_path)
        return sub_paths

    def prune_paths(self, paths: tp.List[tp.List]):
        paths = sorted(paths, key=len, reverse=True)
        used_nodes = set()
        final_paths = []
        for pre_path in paths:
            sub_paths = self._split_path_on_seen_nodes(pre_path, used_nodes)
            for path in sub_paths:
                used_nodes |= {node_side.node_id for node_side in path}
                final_paths.append(self._split_path(path))
        return [ContigPath.from_node_sides(path) for path in final_paths]

    #return paths

    def run(self):
        matrix = self._distance_matrix.data  # +noise
        noise = np.random.rand(*matrix.shape)/10
        noise = noise.T*noise
        matrix = matrix+noise
        assert np.all(matrix.T==matrix)
        row_idx, col_idx = linear_sum_assignment(matrix)
        from_sides = [NodeSide.from_numeric_index(i) for i in row_idx]
        to_sides = [NodeSide.from_numeric_index(i) for i in col_idx]
        side_dict = dict(zip(from_sides, to_sides))
        paths = []
        while len(side_dict):
            paths.append(self.find_one_contig(side_dict))
        return self.prune_paths(paths)

def best_path(distance_matrix: DirectedDistanceMatrix) -> tp.List[ContigPath]:
    noise = np.arange(len(distance_matrix.data))/1000
    noise = noise[:, None]*noise
    matrix = distance_matrix.data  # +noise
    matrix = np.pad(matrix, [(0, 2), (0, 2)], constant_values = np.min(matrix)-1)
    np.fill_diagonal(matrix, np.inf)
    row_idx, col_idx = linear_sum_assignment(matrix)
    from_sides = [NodeSide.from_numeric_index(i) for i in row_idx]
    to_sides = [NodeSide.from_numeric_index(i) for i in col_idx]

    edges = [Edge.from_numeric_index(idx)
             for idx in zip(row_idx[:-2], col_idx[:-2])]
    cur_side = to_sides[-2]
    side_dict = dict(zip(from_sides, to_sides))
    ordered_edges = []
    n_sides = len(from_sides)-2
    used_edges = np.zeros(len(edges))
    node_sides = []
    while True:
        node_sides.append(cur_side)
        cur_side = cur_side.other_side()
        node_sides.append(cur_side)
        next_side = side_dict[cur_side]
        if next_side.numeric_index >= n_sides:
            break
        #ordered_edges.append(Edge(cur_side, next_side))
        cur_side = next_side
    return [ContigPath.from_node_sides(node_sides)]
