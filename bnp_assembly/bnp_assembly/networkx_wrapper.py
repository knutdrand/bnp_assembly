import logging

import networkx as nx

from .distance_matrix import DirectedDistanceMatrix
from .graph_objects import NodeSide, Edge
from .contig_graph import ContigPath
import numpy as np
# from path_finding import PathFinder


class NetworkXContigGraph:
    @classmethod
    def from_distance_matrix(cls, distance_matrix):
        g = nx.Graph()
        data = distance_matrix.data
        assert np.all(~np.isnan(data))
        data_points = [(i, j, -d) for i, row in enumerate(data) for j, d in enumerate(row) if not np.isinf(d)]
        #print(data_points)
        #data_points = [(i, j, -d) for i, row in enumerate(data) for j, d in enumerate(row)]
        logging.info(f'Adding {len(data_points)} edges to NetworkXContigGraph..')
        assert len(data_points) >= 0
        assert all(~np.isnan(dp[2]) for dp in data_points)
        g.add_weighted_edges_from(data_points)
        return g


class PathFinder:
    def __init__(self, distance_matrix: DirectedDistanceMatrix, node_paths=None):
        self._graph = NetworkXContigGraph.from_distance_matrix(distance_matrix)
        self._distance_matrix = distance_matrix

    def get_side_dict(self):
        max_matching = nx.max_weight_matching(self._graph, maxcardinality=True)
        print("Max matching:")
        edges = {NodeSide.from_numeric_index(i): NodeSide.from_numeric_index(j)
                for pair in max_matching for i, j in (pair, pair[::-1])}
        # sort by node id
        edges = {n1: n2 for n1, n2 in sorted(edges.items(), key=lambda x: x[0].node_id)}
        for n1, n2 in edges.items():
            print(f"{n1} -> {n2}")
        return edges

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
            print("Split path: ", path)
            final_paths.append(path)
            seen_sides.add(path[0])
        return final_paths

    def run(self):
        side_dict = self.get_side_dict()
        paths = []
        while len(side_dict):
            new_path = self.find_one_contig(side_dict)
            paths.append(new_path)
            print(f"Found path: {new_path}")
        paths = self.prune_paths(paths)
        return [ContigPath.from_node_sides(path) for path in paths]
