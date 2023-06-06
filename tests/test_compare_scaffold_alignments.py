import pytest
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.agp import ScaffoldAlignments


@pytest.fixture
def true_scaffold():
    return ScaffoldAlignments.from_entry_tuples([
        ('scaffold_1', 0, 100, 'contig_1', 0, 20, '+'),
        ('scaffold_1', 20, 100, 'contig_2', 0, 80, '+'),
        ('scaffold_2', 0, 100, 'contig_3', 0, 100, '+')])


@pytest.fixture
def estimated_scaffold():
    return ScaffoldAlignments.from_entry_tuples([
        ('scaffold_1', 0, 120, 'contig_1', 0, 20, '+'),
        ('scaffold_1', 20, 100, 'contig_2', 0, 80, '+'),
        ('scaffold_1', 100, 200, 'contig_3', 0, 100, '+')])


def test_edge_recall(true_scaffold, estimated_scaffold):
    comparison = ScaffoldComparison(estimated_scaffold, true_scaffold)
    assert comparison.edge_recall() == 1.0
