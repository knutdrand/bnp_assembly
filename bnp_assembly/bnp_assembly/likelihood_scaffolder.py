from collections import Counter
from functools import lru_cache

from bnp_assembly.interface import ScaffolderInterface

#P(d>a and d<a+b) = prod_i(C[a_i+b]-C[a_i]) = prod_i(C[a_i+b]) - prod_i(C[a_i])
#lP = log(C[a+b]-C[a])


class LikelihoodScaffolder(ScaffolderInterface):
    '''
    P(loc_a, loc_b | (node_x, node_y) in edges) =
        P_C(D=d(loc_a, loc_b| (node_x, node_y) in edges))*[d(loc_a, loc_b)]/G if {loc_a.node, loc_b.node} == {node_x, node_y}
        (G-size(node_x)-size(node_y))/ if {loc_a.node, loc_b.node} != {node_x, node_y}
        size(node_x)/G if (loc_a.node == node_x) and (loc_b.node != node_y)

    P(loc_a, loc_b | (node_x, node_y) not in edges) =
          size(node_x)*size(node_y)/G^2 if {loc_a.node, loc_b.node} == {node_x, node_y}
    '''
    def __init__(self, contig_dict, cumulative_distribution: CumulativeDistribution):
        super().__init__(contig_dict)
        self._cumulative_distribution = cumulative_distribution
        self._edge_likelihoods = Counter()
        self._node_counter = Counter()


