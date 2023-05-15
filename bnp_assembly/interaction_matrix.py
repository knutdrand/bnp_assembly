import plotly.express as px
import numpy as np
from .datatypes import GenomicLocationPair

class InteractionMatrix:

    def __init__(self, data, genome_context, bin_size):
        self._data = data
        self._genome_context = genome_context
        self._bin_size = bin_size

    @property
    def data(self) -> np.ndarray:
        return self._data

    @classmethod
    def from_locations_pair(cls, locations_pair: GenomicLocationPair, bin_size=1):
        genome_context = locations_pair.a.genome_context
        assert (len(locations_pair.a.data) == len(locations_pair.b.data)), (len(locations_pair.a.data), len(locations_pair.b.data))
        global_offset = genome_context.global_offset
        total_size = global_offset.total_size
        global_pair = tuple(global_offset.from_local_coordinates(l.chromosome, l.position)//bin_size
                            for l in (locations_pair.a, locations_pair.b))
        size = (global_offset.total_size()+bin_size-1)//bin_size
        print((global_offset.total_size()+bin_size-1), bin_size, global_offset.total_size(), size)
        matrix = np.zeros((size, size))
        print(global_pair)
        np.add.at(matrix, global_pair, 1)
        np.add.at(matrix, global_pair[::-1], 1)
        return cls(matrix, genome_context, bin_size)

    def _transform(self, data):
        return np.log2(data+1)

    def plot(self):
        go = self._genome_context.global_offset
        fig = px.imshow(self._transform(self._data))
        names = go.names()
        offsets=go.get_offset(names)//self._bin_size
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals=offsets, ticktext=names),
                          yaxis = dict(tickmode = 'array', tickvals=offsets, ticktext=names))
        return fig


class SplitterMatrix(InteractionMatrix):

    def normalize_diagonals(self, max_offset)->'SplitterMatrix':
        # copy = self._data.copy()
        copy = self._data+0.01
        means = {i: np.median(np.diagonal(copy, offset=i))
                 for i in range(1, max_offset)}

        for x in range(len(copy)):
            for i in range(1, max_offset):
                y = x-i
                if y >= 0:
                    copy[x, y] /= means[i]
                y = x+i
                if y < len(copy):
                    copy[x, y] /= means[i]
        return SplitterMatrix(copy, self._genome_context, self._bin_size)

    def get_triangle_score(self, bin_n, max_offset):
        score = 0
        n = 0
        for i in range(1, max_offset):
            x = bin_n-i
            if x<0:
                continue
            for j in range(1, i+1):
                y = bin_n+j
                if y >= len(self.data):
                    continue
                score += self.data[x, y]
                n += 1
        if n == 0:
            return 1
        return score/n
