from itertools import product

import numpy as np

from bnp_assembly.graph_objects import Edge, NodeSide


class EdgeCounts:
    def __init__(self, n_nodes):
        self._counts = np.zeros((4, n_nodes, n_nodes))
        self._n_nodes = n_nodes

    def _orientation_idx(self, orientation_a, orientation_b):
        return (orientation_a == 'l') + 2 * (orientation_b == 'l')

    def _edge_idx(self, edge: Edge):
        orientation_idx = self._orientation_idx(edge.from_node_side.side, edge.to_node_side.side)
        return orientation_idx, edge.from_node_side.node_id, edge.to_node_side.node_id

    def keys(self):
        return (Edge(NodeSide(id_a, dir_a), NodeSide(id_b, dir_b))
                for dir_a, dir_b, id_a, id_b in
                product('lr', 'lr', range(self._n_nodes), range(self._n_nodes)))

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, edge: Edge):
        return self._counts[self._edge_idx(edge)]

    def __setitem__(self, edge: Edge, value: float):
        self._counts[self._edge_idx(edge)] = value
