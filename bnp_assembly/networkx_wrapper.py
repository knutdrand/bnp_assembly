import networkx as nx
from .graph_objects import NodeSide
from .contig_graph import ContigPath
# from path_finding import PathFinder

class NetworkXContigGraph:
    @classmethod
    def from_distance_matrix(cls, distance_matrix):
        g = nx.Graph()
        data = distance_matrix.data
        g.add_weighted_edges_from([
            (i, j, -d) for i, row in enumerate(data) for j, d in enumerate(row)])
        return g

class PathFinder:
    def __init__(self, distance_matrix, node_paths=None):
        self._graph = NetworkXContigGraph.from_distance_matrix(distance_matrix)
        self._distance_matrix = distance_matrix

    def get_side_dict(self):
        max_matching = nx.max_weight_matching(self._graph)
        return {NodeSide.from_numeric_index(i): NodeSide.from_numeric_index(j)
                for pair in max_matching for i, j in (pair, pair[::-1])}

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

    def split_path(self, path):
        edges = [Edge(*pair) for pair in zip(path[1::2], path[2::2])] + [Edge(path[-1], path[0])]
        scores = [self._distance_matrix[edge] for edge in edges]
        i = np.argmax(scores)
        cut_idx = 2*(i+1)
        path =  path[cut_idx:] + path[:cut_idx]
        return path

    def prune_paths(self, paths):
        seen_sides = set()
        final_paths = []
        for path in paths:
            if any(node_side in seen_sides for node_side in path):
                continue
            path = self.split_path(path)
            final_paths.append(path)
            seen_sides.add(path[0])
        return final_paths

    def run(self):
        side_dict = self.get_side_dict()
        paths = []
        while len(side_dict):
            paths.append(self.find_one_contig(side_dict))
        paths = self.prune_paths(paths)
        return [ContigPath.from_node_sides(path) for path in paths]
