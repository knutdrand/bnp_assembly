import scipy.stats

from .location import LocationPair, Location
from .contig_map import ScaffoldMap
from .scaffold_splitting import count_possible_edge_pairs, count_edge_overlaps
from .datatypes import GenomicLocationPair
from bionumpy.genomic_data import Genome, GenomicLocation
import numpy as np
from .interaction_matrix import SplitterMatrix, SplitterMatrix2
from .dynamic_bin_distance_matrix import InteractionMatrixFactory
from .plotting import px
from .distance_distribution import calculate_distance_distritbution, distance_dist
# import plotly.express as _px
from .scaffold_splitting.binned_bayes import Yahs

_px = px(name='splitting')
class ScaffoldSplitter:
    def __init__(self, contig_dict, bin_size):
        self._contig_dict = contig_dict
        self._bin_size = bin_size
        self._genome = Genome.from_dict({'0': sum(self._contig_dict.values())})

    def _get_global_location(self, contig_path, locations_pair):
        scaffold_map = ScaffoldMap(contig_path, self._contig_dict)
        global_a = scaffold_map.translate_locations(locations_pair.location_a)
        global_b = scaffold_map.translate_locations(locations_pair.location_b)
        gl_a, gl_b = (GenomicLocation.from_fields(self._genome.get_genome_context(),
                                                  ['0'] * len(g), g) for g in (global_a, global_b))
        return GenomicLocationPair(gl_a, gl_b)

    def split(self, contig_path, locations_pair, threshold=0.5):
        global_locations_pair = self._get_global_location(contig_path, locations_pair)
        interaction_matrix = SplitterMatrix.from_locations_pair(global_locations_pair, self._bin_size)
        normalized = interaction_matrix.normalize_diagonals(10)
        offsets = np.cumsum([self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes])[:-1]
        scores = [normalized.get_triangle_score(offset // self._bin_size, 10) for offset in offsets]
        print(scores)
        px('info').histogram(scores).show()
        px('info').bar(scores).show()
        indices = [i for i, score in enumerate(scores) if score < threshold]
        edges = contig_path.edges
        split_edges = [edges[i] for i in indices]
        return contig_path.split_on_edges(split_edges)



class ScaffoldSplitter2:
    def __init__(self, contig_dict, bin_size):
        self._contig_dict = contig_dict
        self._bin_size = bin_size
        self._genome = Genome.from_dict({'0': sum(self._contig_dict.values())})

    def _get_global_location(self, contig_path, locations_pair):
        scaffold_map = ScaffoldMap(contig_path, self._contig_dict)
        global_a = scaffold_map.translate_locations(locations_pair.location_a)
        global_b = scaffold_map.translate_locations(locations_pair.location_b)
        gl_a, gl_b = (GenomicLocation.from_fields(self._genome.get_genome_context(),
                                                  ['0'] * len(g), g) for g in (global_a, global_b))
        return GenomicLocationPair(gl_a, gl_b)

    def _create_matrix(self, contig_path, locations_pair):
        global_locations_pair = self._get_global_location(contig_path, locations_pair)
        interaction_matrix = SplitterMatrix2.from_locations_pair(global_locations_pair, self._bin_size)
        normalized = interaction_matrix.normalize_matrix()  # .normalize_diagonals(n_bins)
        normalized.plot().show()
        return normalized

    def split(self, contig_path, locations_pair, threshold=0.5, n_bins=20):
        normalized = self._create_matrix(contig_path, locations_pair)
        return self._split_on_matrix(contig_path, normalized, threshold, n_bins)

    def get_edge_bin_ids(self, contig_path):
        return np.cumsum([self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes])[:-1] // self._bin_size

    def _split_on_matrix(self, contig_path, normalized, threshold, n_bins):
        offsets = self.get_edge_bin_ids()
        scores = [normalized.get_triangle_score(offset, n_bins) for offset in offsets]
        q = np.quantile(scores, 0.7)
        threshold = threshold * q
        print(scores)
        print(threshold, q)
        # px('info').histogram(scores).show()
        px('info').bar(y=scores, x=[str(e) for e in contig_path.edges]).show()
        # px('info').bar(scores).show()
        indices = [i for i, score in enumerate(scores) if score < threshold]
        edges = contig_path.edges
        split_edges = [edges[i] for i in indices]
        return contig_path.split_on_edges(split_edges)


class ScaffoldSplitter3(ScaffoldSplitter2):
    def _get_oriented_offsets(self, locations, orientation_dict):
        return Location.from_entry_tuples([(loc.contig_id,
                                            loc.offset if orientation_dict[int(loc.contig_id)] == '+' else
                                            self._contig_dict[int(loc.contig_id)] - loc.offset - 1) for loc in
                                           locations])

    # contig_id, offset, orientation_dict):

    def get_edge_bin_ids(self, *args, **kwargs):
        return self._factory.get_edge_bin_ids()[1:]

    def split(self, contig_path, locations_pair, threshold=0.5, n_bins=40):
        orientation_dict = {dn.node_id: dn.orientation for dn in contig_path.directed_nodes}
        oriented_locations_pair = LocationPair(*(self._get_oriented_offsets(locations, orientation_dict)
                                                 for locations in
                                                 (locations_pair.location_a, locations_pair.location_b)))
        contig_dict = {dn.node_id: self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes}
        self._factory = InteractionMatrixFactory(contig_dict, self._bin_size)
        interaction_matrix = self._factory.create_from_location_pairs(oriented_locations_pair)
        matrix = interaction_matrix.normalize_matrix()
        matrix.plot().show()
        return self._split_on_matrix(contig_path, matrix, threshold, n_bins)


class LinearSplitter(ScaffoldSplitter):
    def __init__(self, contig_dict, threshold):
        self._contig_dict = contig_dict
        self._genome = Genome.from_dict({'0': sum(self._contig_dict.values())})
        self._threshold = threshold

    def _adjust_counts(self, contig_path, edge_counts, boundry_distance_to_weight):
        node_sizes = np.array([self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes])
        distance_to_start = np.cumsum(node_sizes)[:-1]
        distance_to_end = np.cumsum(node_sizes[::-1])[::-1][1:]
        distance_to_edge = np.minimum(distance_to_end, distance_to_start)
        weights = boundry_distance_to_weight[np.minimum(distance_to_edge, boundry_distance_to_weight.size - 1)]
        return edge_counts / weights

    def _split_once(self, contig_path, edge_counts, boundry_distance_to_weight):
        if len(contig_path.directed_nodes) == 1:
            return [contig_path]
        scores = self._adjust_counts(contig_path, edge_counts, boundry_distance_to_weight)
        i = np.argmin(scores)
        if scores[i] > self._threshold:
            return [(contig_path, edge_counts)]
        edge = contig_path.edges[i]
        print(edge)
        return contig_path.split_on_edges([edge])

    def _calculate_boundry_weights(self, location_pair):
        F = distance_dist(location_pair, self._contig_dict)
        F = F[:100000]
        window_size = F.size
        f = lambda x: np.where(x < F.size, F[x], 1)
        p_left_is_matched_in_right = (1 - F[np.arange(window_size)])
        boundry_distance_to_weight = np.cumsum(p_left_is_matched_in_right)
        boundry_distance_to_weight /= boundry_distance_to_weight[-1]
        return boundry_distance_to_weight

    def _get_edge_counts(self, contig_path, locations_pair):
        directed_nodes = contig_path.directed_nodes
        index_dict = {dn.node_id: i for i, dn in enumerate(directed_nodes)}
        start_array = np.zeros(len(directed_nodes))
        end_array = np.zeros(len(directed_nodes))

        global_locations_pair = self._get_global_location(contig_path, locations_pair)
        distance = np.abs(global_locations_pair.a.position - global_locations_pair.b.position)
        mask = distance <= self._window_size
        for a, b, m in zip(locations_pair.location_a, locations_pair.location_b, mask):
            if not m:
                continue
            indices = tuple(index_dict[int(location.contig_id)] for location in (a, b))
            first, second = (min(indices), max(indices))
            if first == second:
                continue
            start_array[first] += 1
            end_array[second] += 1

        start_count = np.cumsum(start_array)
        end_count = np.cumsum(end_array)
        return dict(zip(contig_path.edges, (start_count - end_count)[:-1]))

    def _get_counts_for_path(self, contig_path, edge_counts_dict):
        return np.array([edge_counts_dict[e] for e in contig_path.edges])
        return np.array([self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes])

    def iterative_split(self, contig_path, location_pair):
        boundry_distance_to_weight = self._calculate_boundry_weights(location_pair)
        edge_counts_dict = self._get_edge_counts(contig_path, location_pair)
        edge_counts = self._get_counts_for_path(contig_path, edge_counts_dict)
        px('info').bar(y=edge_counts, x=[str(e) for e in contig_path.edges]).show()
        unfinished = [(contig_path, edge_counts)]
        finished = []
        i = 0
        self._threshold = np.quantile(edge_counts, 0.70) * self._threshold
        while len(unfinished):
            if i > 1000:
                assert False, unfinished
            i += 1
            contig_path, edge_counts = unfinished.pop()
            split_paths = self._split_once(contig_path, edge_counts, boundry_distance_to_weight)
            if len(split_paths) == 1:
                finished.append(contig_path)
            else:
                # px('info').bar(y=edge_counts, x=[str(e) for e in contig_path.edges]).show()

                unfinished += [(cp, self._get_counts_for_path(cp, edge_counts_dict)) for cp in split_paths]
        return list(finished)[::-1]


class LinearSplitter2(LinearSplitter):
    def __init__(self, contig_dict, contig_path, window_size=30000):
        self._contig_dict = contig_dict
        self._genome = Genome.from_dict({'0': sum(self._contig_dict.values())})
        self._contig_path = contig_path
        self._scaffold_map = ScaffoldMap(contig_path, self._contig_dict)
        self._edge_indices = np.cumsum([self._contig_dict[int(dn.node_id)] for dn in contig_path.directed_nodes])[:-1]
        assert self._edge_indices[0] > 0 and self._edge_indices[-1] < sum(self._contig_dict.values())
        self._window_size = window_size

    def _get_all_mapped_locations(self, location_pair):
        locations = np.concatenate([self._scaffold_map.translate_locations(locations)
                                    for locations in (location_pair.location_a, location_pair.location_b)])
        return np.sort(locations)

    def split(self, location_pair):
        F = distance_dist(location_pair, self._contig_dict)
        window_size = min(F.size - 1, self._window_size)

        locations = self._get_all_mapped_locations(location_pair)
        expected = []
        alpha = 0.3

        for idx in self._edge_indices:
            first, mid, last = np.searchsorted(locations, [idx - window_size, idx, idx + window_size])
            print(idx, locations[first], locations[mid], locations[last])
            pL = np.sum(F[window_size] - F[idx - locations[first:mid]])
            pR = np.sum(F[window_size] - F[locations[mid:last] - idx])
            expected.append(pL * pR + alpha)

        edge_counts = self._get_edge_counts(self._contig_path, location_pair)
        scores = [(count + alpha / 2) / expected for count, expected in zip(edge_counts.values(), expected)]
        threshold = 0.3 * np.quantile(scores, 0.7)
        print(scores)
        _px.bar(y=list(edge_counts.values()), x=[str(e) for e in self._contig_path.edges]).show()
        _px.bar(y=expected, x=[str(e) for e in self._contig_path.edges]).show()
        _px.bar(y=scores, x=[str(e) for e in self._contig_path.edges]).show()
        split_edges = [edge for score, edge in zip(scores, self._contig_path.edges) if score < threshold]
        return self._contig_path.split_on_edges(split_edges)


class LinearSplitter3(LinearSplitter2):
    def split(self, location_pair):
        noise_factor = 0.5
        window_size = self._window_size
        locations_pair = [self._scaffold_map.translate_locations(locations)
                          for locations in (location_pair.location_a, location_pair.location_b)]
        locations = np.sort(np.concatenate(locations_pair))
        assert np.all(locations[1:] >= locations[:-1])
        print(locations[-10:], self._edge_indices, self._contig_dict)
        n_locations = len(locations)
        possible_edge_pairs = np.asarray(count_possible_edge_pairs(locations, self._edge_indices, window_size)) * 2
        total_possible_pairs = (n_locations ** 2)
        sampled_pairs = len(locations_pair[0])
        _px.density_heatmap(x=locations_pair[0], y=locations_pair[1], nbinsx=100, nbinsy=100).show()
        # _px.scatter(*locations_pair, ).show()
        print(possible_edge_pairs, total_possible_pairs, sampled_pairs, n_locations)
        expected = np.array(possible_edge_pairs) / total_possible_pairs * sampled_pairs * noise_factor
        edge_values = np.asarray(count_edge_overlaps(*locations_pair, self._edge_indices, window_size))
        print(expected)
        print(edge_values)
        p_values = scipy.stats.poisson.sf(edge_values - 1, expected)
        threshold = 0.05
        _px.scatter(x=expected, y=edge_values, labels={'x': 'expected', 'y': 'observed'},
                    title='expected vs observed').show()
        _px.bar(y=p_values, x=[str(e) for e in self._contig_path.edges]).show()
        split_edges = [edge for p_value, edge in zip(p_values, self._contig_path.edges) if p_value > threshold]
        return self._contig_path.split_on_edges(split_edges)

class YahsSplitter(ScaffoldSplitter3):
    matrix_class = Yahs
    def __init__(self, contig_dict, bin_size):
        super().__init__(contig_dict, bin_size)
        self._bin_size = min(bin_size, max(contig_dict.values())//2)

    def split(self, contig_path, locations_pair, threshold=0.5):
        orientation_dict = {dn.node_id: dn.orientation for dn in contig_path.directed_nodes}
        oriented_locations_pair = LocationPair(*(self._get_oriented_offsets(locations, orientation_dict)
                                                 for locations in
                                                 (locations_pair.location_a, locations_pair.location_b)))
        contig_dict = {dn.node_id: self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes}
        factory = InteractionMatrixFactory(contig_dict, self._bin_size)
        interaction_matrix = factory.create_from_location_pairs(oriented_locations_pair)
        matrix = interaction_matrix.data
        offsets = factory.get_edge_bin_ids()+[len(matrix)]
        start_stop_dict = {i: (start, stop) for i, (start, stop) in
                           enumerate(zip(offsets[:-1], offsets[1:]))}
        assert all([start < stop for start, stop in start_stop_dict.values()])
        yahs = self.matrix_class(matrix, start_stop_dict)
        yahs.save('matrix.npz')
        np.save('contig_path.npy', [dn.node_id for dn in contig_path.directed_nodes])
        scores = yahs.score_vector()
        yahs.plot()
        _px.bar(y=scores, x=[str(e) for e in contig_path.edges]).show()
        indices = [i for i, score in enumerate(scores) if score < np.log(threshold)]
        edges = contig_path.edges
        split_edges = [edges[i] for i in indices]
        return contig_path.split_on_edges(split_edges)

    def _split(self, contig_path, locations_pair, threshold=0.5):
        global_locations_pair = self._get_global_location(contig_path, locations_pair)
        interaction_matrix = SplitterMatrix.from_locations_pair(global_locations_pair, self._bin_size)
        matrix = interaction_matrix.data
        sizes = [self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes]
        offsets = np.cumsum([0] + sizes)
        start_stop_dict = {dn.node_id: (start, stop) for (dn, start, stop) in
                           zip(contig_path.directed_nodes, offsets[:-1], offsets[1:])}
        assert all([start < stop for start, stop in start_stop_dict.values()])
        yahs = Yahs(matrix, start_stop_dict)
        scores = yahs.score_vector()
        _px.bar(y=scores, x=[str(e) for e in contig_path.edges]).show()
        indices = [i for i, score in enumerate(scores) if score < threshold]
        edges = contig_path.edges
        split_edges = [edges[i] for i in indices]
        return contig_path.split_on_edges(split_edges)

