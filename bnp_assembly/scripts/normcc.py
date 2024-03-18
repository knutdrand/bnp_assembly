import scipy
import numpy as np
from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.graph_objects import NodeSide, Edge
import plotly.express as px
from bnp_assembly.make_scaffold import join_all_contigs

m = scipy.sparse.load_npz("test3/Normalized_contact_matrix.npz")
m = m.toarray()

d = DirectedDistanceMatrix(len(m))
for i in range(len(m)):
    for j in range(len(m)):
        for o1 in 'lr':
            for o2 in 'lr':
                edge =  Edge(NodeSide(i, o1), NodeSide(j, o2))
                val = m[i, j]
                assert not np.isnan(val)
                assert not np.isinf(val)
                if i == j:
                    m[i, j] = 0
                else:
                    d[edge] = m[i, j]


#d.invert()
print(d.data)
plot = d.plot(px=px)
plot.show()

path = join_all_contigs(d)

print(path)