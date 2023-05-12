from .location import LocationPair, Location
from .contig_map import ScaffoldMap
from .datatypes import GenomicLocationPair
from bionumpy.genomic_data  import Genome, GenomicLocation
import numpy as np
from .interaction_matrix import SplitterMatrix

class ScaffoldSplitter:
    def __init__(self, contig_dict, bin_size):
        self._contig_dict = contig_dict
        self._bin_size = bin_size
        self._genome = Genome.from_dict({'0': sum(self._contig_dict.values())})

    def split(self, contig_path, locations_pair, threshold=0.5):
        scaffold_map = ScaffoldMap(contig_path, self._contig_dict)
        global_a = scaffold_map.translate_locations(locations_pair.location_a)
        global_b = scaffold_map.translate_locations(locations_pair.location_b)
        gl_a, gl_b  = (GenomicLocation.from_fields(self._genome.get_genome_context(),
                                           ['0']*len(g), g) for g in (global_a, global_b))
                                                   
        global_locations_pair = GenomicLocationPair(gl_a, gl_b)
        interaction_matrix = SplitterMatrix.from_locations_pair(global_locations_pair, self._bin_size)
        normalized = interaction_matrix.normalize_diagonals(10)
        offsets = np.cumsum([self._contig_dict[dn.node_id] for dn in contig_path.directed_nodes])[:-1]
        scores =  [normalized.get_triangle_score(offset//self._bin_size, 10) for offset in offsets]
        indices = [i for i, score in enumerate(scores) if score<threshold]
        edges = contig_path.edges
        split_edges = [edges[i] for i in indices]
        return contig_path.split_on_edges(split_edges)
