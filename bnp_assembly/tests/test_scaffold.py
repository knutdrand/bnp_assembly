import pytest

from bnp_assembly.contig_graph import ContigPath, DirectedNode
from bnp_assembly.scaffolds import Scaffolds, Scaffold


@pytest.fixture
def contig_path():
    return ContigPath.from_directed_nodes([DirectedNode(0, '+'), DirectedNode(1, '-')])


@pytest.fixture
def translation_dict():
    return {0: 'chr1', 1: 'chr2'}


def test_from_contig_path(contig_path, translation_dict):
    scaffold = Scaffold.from_contig_path(contig_path, translation_dict)
    assert scaffold.name is None
    assert scaffold.to_contig_path(translation_dict) == contig_path
