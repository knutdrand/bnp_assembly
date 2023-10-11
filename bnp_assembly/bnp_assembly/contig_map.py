from .contig_graph import ContigPath
from bionumpy.genomic_data import Genome
from .location import Location
import numpy as np
import typing as tp


class ScaffoldMap:
    def __init__(self, contig_path: ContigPath, contig_dict):
        self._contig_path = contig_path
        self._contig_dict = contig_dict
        directed_nodes = contig_path.directed_nodes
        offsets = np.cumsum([contig_dict[directed_node.node_id] for directed_node in directed_nodes])
        offsets = np.insert(offsets, 0, 0)
        self._offset_dict = {directed_node.node_id: offset for directed_node, offset in zip(directed_nodes, offsets)}
        self._direction_dict = {dn.node_id: dn.orientation for dn in directed_nodes}

    def translate_locations(self, locations: Location):
        offsets = self._offset_dict
        return np.array([self.translate_location(loc) for loc in locations])

    def translate_location(self, location: Location.single_entry):
        node_id = int(location.contig_id)
        offset = self._offset_dict[node_id]
        if self._direction_dict[node_id] == '-':
            local_offset = self._contig_dict[node_id] - location.offset - 1
        else:
            local_offset = location.offset
        return offset + local_offset


class VectorizedScaffoldMap(ScaffoldMap):
    def __init__(self, contig_path, contig_dict):
        super().__init__(contig_path, contig_dict)
        self._offset_array = np.array([self._offset_dict[dn.node_id] for dn in contig_path.directed_nodes])
        self._is_reverse = np.array([dn.orientation == '-' for dn in contig_path.directed_nodes])

    def translate_locations(self, locations):
        local_offset = np.where(self._is_reverse[locations.contig_id],
                                self._contig_dict[locations.contig_id] - locations.offset - 1, locations.offset)
        return self._offset_array[locations.contig_id] + local_offset


class ContigMap:

    def __init__(self, original_genome: Genome, new_genome: Genome, offsets):
        self._original_genome = original_genome
        self._new_genome = new_genome

    def translate_coordinates(self, original_coordinates):
        pass

    @classmethod
    def from_original_and_paths(cls, genome: Genome, path_lists: tp.List[ContigPath]):
        pass
