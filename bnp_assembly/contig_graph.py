import numpy as np


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
        to_array  = lambda d: np.array([[d[a][b] for a in nodes] for b in nodes])
        return cls(nodes, *(to_array(d) for d in (rl_edges, rr_edges, ll_edges)))
                   
