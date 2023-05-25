import plotly.express as px
import numpy as np
from .datatypes import GenomicLocationPair


class InteractionMatrix:
    def __init__(self, data: object, genome_context: object, bin_size: object) -> object:
        self._data = data
        self._genome_context = genome_context
        self._bin_size = bin_size

    @property
    def data(self) -> np.ndarray:
        return self._data

    def normalize_rows(self):
        return self.__class__(self._data/np.sum(self._data, axis=-1, keepdims=True),
                              self._genome_context,
                              self._bin_size)

    def normalize_matrix(self):
        norm1 = (np.mean(self._data, axis=-1, keepdims=True))
        norm2 = np.mean(self._data, axis=0, keepdims=True)
        mask = (norm1*norm2) == 0
        new_data = np.where(mask, 0, self._data/(norm1*norm2))
        return self.__class__(new_data,
                              self._genome_context,
                              self._bin_size)


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
        offsets = go.get_offset(names)//self._bin_size
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



class SplitterMatrix2(InteractionMatrix):
    def normalize_diagonals(self, max_offset)->'SplitterMatrix2':
        # copy = self._data.copy()
        copy = self._data.astype(float)
        # adding + 1 before taking median to avoid zero median
        means = {i: np.quantile(np.diagonal(copy, offset=i) + 1, 0.8)
                 for i in range(1, max_offset)}

        for x in range(len(copy)):
            for i in range(1, max_offset):
                y = x-i
                if y >= 0:
                    copy[x, y] = min(copy[x, y], means[i])
                y = x+i
                if y < len(copy):
                    copy[x, y] = min(copy[x, y], means[i])
                    # copy[x, y] /= means[i]
        copy[np.isnan(copy)] = 0

        assert np.all(~np.isnan(copy))

        return SplitterMatrix2(copy, self._genome_context, self._bin_size)

    def get_triangle_score(self, bin_n, max_offset):
        return get_weighted_triangle_score(self.data, bin_n, max_offset)  # , ignore_positions=np.sum(self._data, axis=-1)==0)


def get_weighted_triangle_score(matrix, bin_n, max_offset, ignore_positions=None):
    subset = matrix[bin_n:bin_n+max_offset, bin_n-max_offset:bin_n].copy()
    if subset.size > 0:
        subset[0, 0] = 0
    #px.imshow(subset, range_color=(0, 2000), title="BIN number: " + str(bin_n)).show()
    score = 0
    n = 0
    if ignore_positions is None:
        ignore_positions = np.zeros(len(matrix), dtype=bool)
    for i in range(0, max_offset+1):
        x = bin_n+i
        if x >= len(matrix):
            continue
        if ignore_positions[x]:
            continue

        for j in range(1, max_offset-i):
            if i == 0 and j == 0:
                continue
            y = bin_n-j
            if y < 0:
                continue
            if ignore_positions[y]:
                continue
            score += matrix[x, y] # /(i+1)
            n += 1
    if n == 0:
        return 1
    return score/n




def get_triangle_score(matrix, bin_n, max_offset):
    score = 0
    n = 0
    for i in range(0, max_offset+1):
        x = bin_n+i
        if x >= len(matrix):
            continue
        for j in range(0, max_offset-i+1):
            if i == 0 and j == 0:
                continue
            y = bin_n-j
            if y < 0:
                continue
            score += matrix[x, y] # /(i+1)
            n += 1 # /(i+1)
    if n == 0:
        return 1
    return score# /n
