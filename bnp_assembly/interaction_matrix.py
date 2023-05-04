import plotly.express as px
import numpy as np
from .datatypes import GenomicLocationPair

class InteractionMatrix:

    def __init__(self, data, genome_context):
        self._data = data
        self._genome_context = genome_context

    @property
    def data(self):
        return self._data

    @classmethod 
    def from_locations_pair(cls, locations_pair: GenomicLocationPair):
        genome_context = locations_pair.a.genome_context
        global_offset = genome_context.global_offset
        global_pair = tuple(global_offset.from_local_coordinates(l.chromosome, l.position)
                            for l in (locations_pair.a, locations_pair.b))
        size = global_offset.total_size()
        matrix=  np.zeros((size, size))
        np.add.at(matrix, global_pair, 1)
        np.add.at(matrix, global_pair[::-1], 1)
        return cls(matrix, genome_context)
