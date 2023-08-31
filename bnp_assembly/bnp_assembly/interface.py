from typing import List

import numpy as np

from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.coordinate_system import CoordinateSystem
from bnp_assembly.graph_objects import Edge
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.plotting import px
from bnp_assembly.splitting import split_on_scores


def weighted_median(values, weights):
    ''' From stackoverflow'''
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


def score_matrix(observed: np.ndarray, expected: np.ndarray, edge: Edge=None)->float:
    """
    Score a matrix of observed counts against a matrix of expected counts.
    This method should take account for both bands of overrepresntation and underrepresentation.

    Parameters
    ----------
    observed
    expected
    edge

    Returns
    -------

    """
    shape = observed.shape
    expected = expected[:shape[0], :shape[1]]
    observed = np.minimum(observed, expected) # Take the minimum here to avoid giving impact to high values
    px(name='splitting').imshow(observed, title=f'truncated{edge}')
    expected_a = np.sum(expected, axis=1)
    expected_b = np.sum(expected, axis=0)
    a = np.sum(observed, axis=1) / expected_a
    b = np.sum(observed, axis=0) / expected_b
    ratio = np.concatenate([a, b]) # These are a lot of column and row sums
    # If we take the average of these, we give impact to high bands
    # Taking the median or a more robust measure should be better
    px(name='splitting').histogram(ratio, nbins=20, title=f'edge {edge}')
    new_expected = np.concatenate([expected_a, expected_b])
    '''
    args = np.argsort(ratio)
    ratio = ratio[args]
    expected = expected[args]
    cut_n = len(ratio)//8
    ratio = ratio[cut_n:-cut_n]
    expected = expected[cut_n:-cut_n]
    '''
    return weighted_median(ratio, new_expected)


class SplitterInterface:
    def __init__(self, contig_dict, location_pairs, contig_path, max_distance=100000, bin_size=1000, threshold=0.2):
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
        self._threshold = threshold

    def count_bins(self, node_id):
        return int(np.ceil(min(self._max_distance, self._contig_dict[node_id]) / self._bin_size))

    def _register_intra_pair(self, location_pair):
        """
        Register a pair of locations that are on the same contig.
        This is used to build the histogram of distances between locations on the same contig.
        Parameters
        ----------
        location_pair

        Returns
        -------

        """
        d = abs(location_pair.location_a.offset // self._bin_size - location_pair.location_b.offset // self._bin_size)
        if d >= self._n_bins * 2:
            return
        self.distance_counts[d] += 1

    def register_location_pair(self, location_pair: LocationPair):
        """
        Register a pair of locations

        Parameters
        ----------
        location_pair

        Returns
        -------

        """
        if location_pair.location_a.contig_id == location_pair.location_b.contig_id:
            return self._register_intra_pair(location_pair)
        if location_pair in self._coordinate_system:
            return self._register_inter_pair(location_pair)

    def _register_inter_pair(self, location_pair: LocationPair):
        """
        Register a pair of locations that are on neighboring contigs.
        Adds a count to the corresponding count matrix

        Parameters
        ----------
        location_pair

        Returns
        -------

        """
        edge, coordinates = self._coordinate_system.location_pair_coordinates(location_pair)
        if any(coord >= self._max_distance for coord in coordinates):
            return
        coordinates = [coord // self._bin_size for coord in coordinates]
        self._node_histograms[edge][coordinates[0], coordinates[1]] += 1

    def split(self) -> List[ContigPath]:
        """
        Split the contig on edges where the count matrices appears to not support an edge.
        Returns
        -------
        """
        for a, b in zip(self._location_pairs.location_a, self._location_pairs.location_b):
            self.register_location_pair(LocationPair(Location.single_entry(int(a.contig_id), int(a.offset)),
                                                     Location.single_entry(int(b.contig_id), int(b.offset))))
        self.plot()
        px(name='splitting').line(self.distance_counts, title='distance counts')
        distance_means = self._normalize_dist_counts(self.distance_counts)
        assert np.all(~np.isnan(distance_means)), distance_means
        expected = np.zeros((self._n_bins, self._n_bins))
        for i in range(self._n_bins):
            for j in range(self._n_bins):
                expected[i, j] = distance_means[i + j + 1]
        assert np.all(~np.isnan(expected)), expected
        # Fill the expected matrix with the expected values according to distance
        px(name='splitting').imshow(expected, title='expected')
        scores = {edge: score_matrix(self._node_histograms[edge], expected, edge) for edge in self._contig_path.edges}
        return split_on_scores(self._contig_path, scores, self._threshold, keep_over=True)

    def plot(self):
        for edge, histogram in self._node_histograms.items():
            px(name='splitting').imshow(histogram, title=f'matrix{edge}')

    def _normalize_dist_counts(self, distance_counts):
        """
        Normalize distance counts by the number of cells that has that distance (i-j)==d

        Parameters
        ----------
        distance_counts

        Returns
        -------

        """
        n_attempts = np.zeros_like(distance_counts)
        for length in self._contig_dict.values():
            k = length // self._bin_size
            for i in range(min(k, n_attempts.size)):
                n_attempts[i] += k - i
        return np.minimum.accumulate(np.where(n_attempts>0, distance_counts / n_attempts, 0))
