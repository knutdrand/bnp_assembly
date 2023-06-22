import numpy as np

from bnp_assembly.coordinate_system import CoordinateSystem
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.plotting import px
from bnp_assembly.splitting import split_on_scores

def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]


class ScaffolderInterface:
    def __init__(self, contig_dict):
        self._contig_dict = contig_dict

    def preprocess(self):
        return NotImplemented

    def register_read_pairs(self, read_pairs: LocationPair):
        return NotImplemented

    def get_distance_matrix(self):
        return NotImplemented

    def get_scaffold(self, location_pairs):
        self.register_read_pairs(location_pairs)
        distance_matrix = self.get_distance_matrix()
        return self.get_scaffold_from_distance_matrix(distance_matrix)

    def get_scaffold_from_distance_matrix(self, distance_matrix):
        pass


class SplitterInterface:
    def __init__(self, contig_dict, location_pairs, contig_path, max_distance=100000, bin_size=1000):
        self._contig_dict = contig_dict
        self._location_pairs = location_pairs
        self._contig_path = contig_path
        assert max_distance % bin_size == 0
        self._max_distance = max_distance
        self._bin_size = bin_size
        self._n_bins = max_distance // bin_size
        self._hist_shapes = {
            edge: (self.count_bins(edge.from_node_side.node_id), self.count_bins(edge.to_node_side.node_id)) for edge in
            contig_path.edges}
        n_bins = self._n_bins
        self._node_histograms = {edge: np.zeros(self._hist_shapes[edge]) for edge in contig_path.edges}
        self._coordinate_system = CoordinateSystem(contig_dict, contig_path.edges)
        self.distance_counts = np.zeros(n_bins * 2)

    def count_bins(self, node_id):
        return int(np.ceil(min(self._max_distance, self._contig_dict[node_id]) / self._bin_size))

    def _register_intra_pair(self, location_pair):
        d = abs(location_pair.location_a.offset // self._bin_size - location_pair.location_b.offset // self._bin_size)
        if d >= self._n_bins * 2:
            return
        self.distance_counts[d] += 1

    def score_matrix(self, observed, expected):
        shape = observed.shape
        expected = expected[:shape[0], :shape[1]]
        observed = np.minimum(observed, expected)
        expected_a = np.sum(expected, axis=1)
        expected_b = np.sum(expected, axis=0)
        a = np.sum(observed, axis=1) / expected_a
        b = np.sum(observed, axis=0) / expected_b
        px(name='splitting').histogram(np.concatenate([a, b]), nbins=20)
        return weighted_median(np.concatenate([a, b]), np.concatenate([expected_a, expected_b]))

    def register_location_pair(self, location_pair):
        if location_pair not in self._coordinate_system:
            self._register_intra_pair(
                location_pair)  # a, b = int(location_pair.location_a.contig_id), int(location_pair.location_b.contig_id)
            return
        edge, coordinates = self._coordinate_system.location_pair_coordinates(location_pair)
        if any(coord >= self._max_distance for coord in coordinates):
            return
        coordinates = [coord // self._bin_size for coord in coordinates]
        self._node_histograms[edge][coordinates[0], coordinates[1]] += 1

    def split(self):
        for a, b in zip(self._location_pairs.location_a, self._location_pairs.location_b):
            self.register_location_pair(LocationPair(Location.single_entry(int(a.contig_id), int(a.offset)),
                                                     Location.single_entry(int(b.contig_id), int(b.offset))))
        self.plot()
        px(name='splitting').line(self.distance_counts, title='distance counts')
        distance_means = self._normalize_dist_counts(self.distance_counts)
        expected = np.zeros((self._n_bins, self._n_bins))
        for i in range(self._n_bins):
            for j in range(self._n_bins):
                expected[i, j] = distance_means[i + j + 1]

        px(name='splitting').imshow(expected, title='expected')
        scores = {edge: self.score_matrix(self._node_histograms[edge], expected) for edge in self._contig_path.edges}
        return split_on_scores(self._contig_path, scores, 0.1, keep_over=True)

    def plot(self):
        for edge, histogram in self._node_histograms.items():
            px(name='splitting').imshow(histogram, title=f'matrix{edge}')

    def _normalize_dist_counts(self, distance_counts):
        n_attempts = np.zeros_like(distance_counts)
        for length in self._contig_dict.values():
            k = length // self._bin_size
            for i in range(min(k, n_attempts.size)):
                n_attempts[i] += k - i
            # n_attempts[:k] += 1# np.arange(k, 0, -1)[:k]
        return np.minimum.accumulate(distance_counts / n_attempts)
