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
        global_offset = genome_context.global_offset
        global_pair = tuple(global_offset.from_local_coordinates(l.chromosome, l.position)//bin_size
                            for l in (locations_pair.a, locations_pair.b))
        size = (global_offset.total_size()+bin_size-1)//bin_size
        matrix =  np.zeros((size, size))
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
