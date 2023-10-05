from typing import Iterable

from .io import PairedReadStream
from .location import LocationPair
from .pre_sampled_dynamic_heatmap_comparison import DynamicHeatmapDistanceFinder


class ReadCountBasedHeatmapDistanceFinder(DynamicHeatmapDistanceFinder):

    def find_scale_funcs(self, reads: Iterable[LocationPair], contig_sizes: np.array):
        pass

    def __call__(self, reads: PairedReadStream, effective_contig_sizes):
        """
        Returns a DirectedDistanceMatrix with "distances" using the dynamic heatmap method
        (comparing heatmaps for contigs against background heatmaps)
        """
        input_data = NumericInputData(effective_contig_sizes, reads)
        assert isinstance(input_data.location_pairs, PairedReadStream), type(input_data.location_pairs)
        dynamic_heatmap_creator = PreComputedDynamicHeatmapCreator(input_data.contig_dict, self._heatmap_config)
        sampled_heatmaps = dynamic_heatmap_creator.create(input_data.location_pairs, n_extra_heatmaps=0)
        heatmap_comparison = HeatmapComparisonRowColumns.from_heatmap_stack(list(sampled_heatmaps.values())[::-1], add_n_extra=3)
        heatmaps = get_dynamic_heatmaps_from_reads(self._heatmap_config, input_data)

        distances = {}
        for edge in get_all_possible_edges(len(input_data.contig_dict)):
            heatmap = heatmaps.get_heatmap(edge)
            plot_name = None
            if edge.from_node_side.node_id == edge.to_node_side.node_id - 1:
                plot_name = f"Searcshorted indexes {edge}"
            distance = heatmap_comparison.get_heatmap_score(heatmap, plot_name=plot_name)
            distances[edge] = distance

            if edge.from_node_side.node_id == edge.to_node_side.node_id - 1:
                px(name="dynamic_heatmaps").imshow(heatmap.array, title=f"Edge heatmap {edge}")
                print(edge, distance)

        DirectedDistanceMatrix.from_edge_dict(len(input_data.contig_dict), distances).plot(
            name="dynamic_heatmap_scores").show()

        return DirectedDistanceMatrix.from_edge_dict(len(effective_contig_sizes), distances)

    def __init__(self):


