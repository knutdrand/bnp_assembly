from .contig_graph import ContigPath
from bionumpy.genomic_data import Genome
import typing as tp
class ContigMap:

    def __init__(self, original_genome: Genome, new_genome: Genome, offsets):
        self._original_genome = original_genome
        self._new_genome = new_genome

    def translate_coordinates(self, original_coordinates):
        pass

    @classmethod
    def from_original_and_paths(cls, genome: Genome, path_lists: tp.List[ContigPath]):
        pass
        
        
