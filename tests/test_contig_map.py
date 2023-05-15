from bionumpy.genomic_data import Genome
from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.contig_map import ScaffoldMap
from bnp_assembly.location import Location
import pytest


@pytest.fixture
def contig_dict():
    return {0: 100, 1: 50}


@pytest.fixture
def contig_path():
    return ContigPath.from_directed_nodes([DirectedNode(0, '+'), DirectedNode(1, '-')])


@pytest.fixture
def scaffold_map(contig_dict, contig_path):
    return ScaffoldMap(contig_path, contig_dict)


def test_map(scaffold_map):
    assert scaffold_map.translate_location(Location.single_entry(0, 5))==5
    assert scaffold_map.translate_location(Location.single_entry(1, 5))==145
    


'''
@pytest.fixture
def original_genome():
    return Genome.from_dict({'A': 10,
                             'B': 20, 
                             'C': 30})


@pytest.fixture
def paths():
    return [ContigPath.from_directed_nodes([DirectedNode('B', '+'),
                                            DirectedNode('A', '-')]),
            ContigPath.from_directed_nodes([DirectedNode('C', '+'),
                                           DirectedNode('B', '-')]
            

'''
