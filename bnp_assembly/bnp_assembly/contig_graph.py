from dataclasses import dataclass
import numpy as np
from more_itertools import chunked
from scipy.optimize import linear_sum_assignment
from .graph_objects import NodeSide, Edge
import typing as tp

@dataclass
class DirectedNode:
    node_id: str
    orientation: str

    def __repr__(self):
        return f'{self.node_id}{self.orientation}'


class ContigPath:
    def __init__(self, node_ids, reverse_mask, node_names=None):
        self._node_ids = np.asarray(node_ids)
        self._reverse_mask = np.asarray(reverse_mask)
        self._node_names = node_names

    @classmethod
    def from_list(cls, directed_nodes, node_name_list: list=None):
        node_names, reverse_mask = zip(*directed_nodes)
        if node_name_list is not None:
            node_ids = [node_name_list.index(name) for name in node_names]
            node_names = node_name_list
            return cls(node_ids, reverse_mask, node_names)
        assert False

    def __repr__(self):
        return f'ContigPath({self.to_list()})'

    def reverse(self):
        return self.__class__(self._node_ids[::-1], (1-self._reverse_mask)[::-1], self._node_names)

    def to_list(self):
        return [(self._node_names[node_id], strand) for node_id, strand in zip(self._node_ids, self._reverse_mask)]

    @classmethod
    def from_directed_nodes(cls, directed_nodes: tp.List[DirectedNode]):
        node_sides = []
        for directed_node in directed_nodes:
            sides = 'lr' if directed_node.orientation == '+' else 'rl'
            node_sides.extend(NodeSide(directed_node.node_id, side) for side in sides)
        return ContigPathSides(node_sides)

    @classmethod
    def from_edges(cls, edges):
        assert len(edges)>0
        sides = [edges[0].from_node_side.other_side()]
        for edge in edges:
            sides.append(edge.from_node_side)
            sides.append(edge.to_node_side)
        sides.append(edges[-1].to_node_side.other_side())
        return ContigPathSides(sides)

    @classmethod
    def from_node_sides(cls, node_sides):
        return ContigPathSides(node_sides)

    def __eq__(self, other):
        assert isinstance(other, ContigPath)
        return self.to_list() == other.to_list() or self.reverse().to_list() == other.to_list()


class ContigPathSides(ContigPath):
    def __init__(self, node_sides):
        self._node_sides = node_sides

    @property
    def directed_nodes(self):
        return [DirectedNode(side_1.node_id, '+' if side_1.side=='l' else '-')
                for side_1, _ in chunked(self._node_sides, 2)]

    def split_on_edge(self, edge: Edge):
        cut_idx = (self.edges.index(edge)+1)*2
        return self.__class__(self._node_sides[:cut_idx]), self.__class__(self._node_sides[cut_idx:])

    def split_on_edges(self, edges: tp.List[Edge]):
        paths = []
        cur_path = self
        for edge in edges:
            first, cur_path = cur_path.split_on_edge(edge)
            paths.append(first)
        paths.append(cur_path)
        return paths

    @property
    def edges(self):
        return [Edge(*pair) for pair in zip(self._node_sides[1::2], self._node_sides[2::2])]

    @property
    def nodes(self):
        return [node_side.node_id for node_side in self._node_sides[::2]]

    @property
    def node_sides(self):
        return self._node_sides

    def reverse(self):
        return self.__class__(self._node_sides[::-1])

    def to_list(self):
        return [(node_side.node_id, int(node_side.side == 'r'))
                for node_side in self._node_sides[::2]]

class ContigPathEdges(ContigPath):
    def __init__(self, edges):
        self._edges = edges

    @property
    def edges(self):
        return self._edges

    @property
    def nodes(self):
        if len(self._edges)==0:
            return []
        return [edge.from_node_side.node_id for edge in self._edges] + [self._edges[-1].to_node_side.node_id]

    def reverse(self):
        return self.__class__([e.reverse() for e in self._edges[::-1]])

    def to_list(self):
        nodes = [(edge.from_node_side.node_id, int(edge.from_node_side.side == 'l'))
                 for edge in self._edges]
        last_side = self._edges[-1].to_node_side
        return nodes + [(last_side.node_id, int(last_side.side == 'r'))]

    def __repr__(self):
        return repr(self.to_list())

    def __eq__(self, other):
        assert isinstance(other, ContigPathEdges)
        return self.to_list() == other.to_list() or self.reverse().to_list() == other.to_list()

class ContigGraph:
    def __init__(self, nodes, rl_edges, rr_edges, ll_edges):
        self._nodes = nodes
        self._rl_edges = rl_edges
        self._rr_edges = rr_edges
        self._ll_edges = ll_edges

    @property
    def nodes(self):
        return self._nodes

    @classmethod
    def from_distance_dicts(cls, rl_edges, rr_edges, ll_edges):
        nodes = list(rl_edges.keys())
        to_array = lambda d: np.array([[d[a][b] for a in nodes] for b in nodes])
        return cls(nodes, *(to_array(d) for d in (rl_edges, rr_edges, ll_edges)))

    def best_path(self):
        n = len(self._nodes)
        distance_matrix = np.zeros((2+2*n, 2+2*n))
        matrices = [[self._ll_edges, self._rl_edges.T],
                    [self._rl_edges, self._rr_edges]]
        for from_dir in (0, 1):
            for to_dir in (0, 1):
                distance_matrix[from_dir*n:from_dir*n+n, to_dir*n:to_dir*n+n] = matrices[from_dir][to_dir]
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix[-2:, -2:] = np.inf
        row_idx, col_idx = linear_sum_assignment(distance_matrix)
        cur_idx = col_idx[-1]
        path = []
        seen = set()
        node_ids = []
        reverse_mask = []
        while cur_idx < 2*n:
            node_id, strand = (cur_idx % n, cur_idx//n)
            if node_id in seen:
                break
            node_ids.append(node_id)
            reverse_mask.append(strand)
            seen.add(node_id)
            path.append((self._nodes[node_id], strand))
            cur_idx = col_idx[(1-strand)*n+node_id]
        return ContigPath(node_ids, reverse_mask, self.nodes)
    # return [(self._nodes[elem % n], elem//n) for elem in path]
