import numpy as np
from scipy.optimize import linear_sum_assignment


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

    def reverse(self):
        return self.__class__(self._node_ids[::-1], (1-self._reverse_mask)[::-1], self._node_names)

    def to_list(self):
        return [(self._node_names[node_id], strand) for node_id, strand in zip(self._node_ids, self._reverse_mask)]

    @classmethod
    def from_edges(cls, edges):
        return ContigPathEdges(edges)


class ContigPathEdges(ContigPath):
    def __init__(self, edges):
        self._edges = edges

    @property
    def edges(self):
        return self._edges

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
        print(distance_matrix)
        row_idx, col_idx = linear_sum_assignment(distance_matrix)
        print(col_idx)
        cur_idx = col_idx[-1]
        path = []
        seen = set()
        node_ids = []
        reverse_mask = []
        while cur_idx < 2*n:
            print(cur_idx)
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
