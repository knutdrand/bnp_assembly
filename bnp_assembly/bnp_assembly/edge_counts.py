from functools import singledispatchmethod
from itertools import product

import numpy as np

from bnp_assembly.graph_objects import Edge, NodeSide


def single_dispatchmethod(args):
    pass


class EdgeCounts:
    def __init__(self, n_nodes):
        self._counts = np.zeros((n_nodes, n_nodes, 2, 2))
        self._n_nodes = n_nodes

    def _orientation_idx(self, orientation):
        return 0 if orientation == 'l' else 1

    def _edge_idx(self, edge: Edge):
        return edge.from_node_side.node_id, edge.to_node_side.node_id, self._orientation_idx(edge.from_node_side.side), self._orientation_idx(edge.to_node_side.side)

    def keys(self):
        return (Edge(NodeSide(id_a, dir_a), NodeSide(id_b, dir_b))
                for dir_a, dir_b, id_a, id_b in
                product('lr', 'lr', range(self._n_nodes), range(self._n_nodes)))

    def values(self):
        return iter(self._counts.ravel())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._counts.ravel())

    @singledispatchmethod
    def __getitem__(self, edge):
        raise NotImplementedError

    @__getitem__.register
    def _(self, edge: Edge):
        return self._counts[self._edge_idx(edge)]

    @__getitem__.register
    def _(self, edge: tuple):
        return self._counts[edge]

    @singledispatchmethod
    def __setitem__(self, edge, value):
        raise NotImplementedError

    @__setitem__.register
    def _(self, edge: tuple, value):
        assert np.all(~np.isnan(value))
        self._counts[edge] = value

    @__setitem__.register
    def _(self, edge: Edge, value):
        assert np.all(~np.isnan(value))
        self._counts.__setitem__(self._edge_idx(edge), value)
