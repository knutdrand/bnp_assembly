from .graph_objects import NodeSide, Edge
import numpy as np
from .plotting import px, DummyPlot, level_dict
import logging
from .plotting import px as px_func


class DirectedDistanceMatrix:
    def __init__(self, n_nodes):
        self.px = px_func(name='joining')
        n_sides = n_nodes*2
        self._matrix = np.zeros((n_sides, n_sides))
        self._fill_infs()

    def _fill_infs(self):
        np.fill_diagonal(self._matrix, np.inf)
        for i in range(len(self._matrix)):
            node_side = NodeSide.from_numeric_index(i)
            self._matrix[Edge(node_side, node_side.other_side()).numeric_index] = np.inf

    @classmethod
    def from_edge_dict(cls, n_nodes, d):
        mat = cls(n_nodes)
        for edge, value in d.items():
            mat[edge] = value
        return mat

    def keys(self):
        all_edges = (Edge.from_numeric_index((i, j)) for i in range(len(self._matrix)) for j in range(len(self._matrix)))
        return (edge for edge in all_edges if edge.from_node_side.node_id != edge.to_node_side.node_id)

    @property
    def data(self):
        return self._matrix

    def __len__(self):
        return len(self._matrix)

    def __setitem__(self, edge: Edge, score: float):
        if edge.from_node_side.node_id == edge.to_node_side.node_id:
            # logging.warning(f'Illegal edge set in distance_matrix: {edge}')
            return
        assert not np.isnan(score)
        self._matrix[edge.numeric_index] = score

    def __getitem__(self, edge: Edge):

        return self._matrix[edge.numeric_index]

    def inversion_plot(self, name):
        n_nodes = len(self)//2
        table = {'from_node': [], 'to_node': [], 'score': [], 'orientation': []}
        for node_id in range(n_nodes-1):
            node_id2  = node_id + 1
            for dir_a in 'rl':
                for dir_b in 'lr':
                    edge = Edge(NodeSide(node_id, dir_a), NodeSide(node_id2, dir_b))
                    table['from_node'].append(node_id)
                    table['to_node'].append(node_id2)
                    table['score'].append(self[edge])
                    table['orientation'].append(f'{dir_a}{dir_b}')
        self.px.line(table, x='from_node', y='score', line_shape='hv', color='orientation', title=f'Inversion plot-{name}')

    def plot(self, level=logging.INFO, name=''):
        n_nodes = len(self)//2
        new_matrix = np.empty((n_nodes, n_nodes))
        max_value = self.data.max()
        matrices = {'{dira}{dirb}'}
        for node_id in range(n_nodes):
            for node_id_2 in range(n_nodes):
                if node_id == node_id_2:
                    new_matrix[node_id, node_id_2] = np.nan
                    continue
                edge = Edge(NodeSide(node_id, 'r'), NodeSide(node_id_2, 'l'))
                new_matrix[node_id, node_id_2] = self[edge]
                edge_r = Edge(NodeSide(node_id, 'l'), NodeSide(node_id_2, 'r'))
                new_matrix[node_id_2, node_id] = self[edge_r]

        #assert np.all(~np.isnan(new_matrix))
        #self.px.array(new_matrix, title="distance_matrix")
        return self.px.imshow(new_matrix, zmax=0, title=name)
        # fig = px(level).imshow(new_matrix)
        #go = self._genome_context.global_offset
        #fig = px.imshow(self._transform(self._data))
        #names = go.names()
        #offsets=go.get_offset(names)//self._bin_sizep
        #fig.update_layout(xaxis = dict(tickmode = 'array', tickvals=offsets, ticktext=names),
        #                  yaxis = dict(tickmode = 'array', tickvals=offsets, ticktext=names))
        # return fig
