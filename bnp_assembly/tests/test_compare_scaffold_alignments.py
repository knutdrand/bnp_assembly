import pytest
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.agp import ScaffoldAlignments


@pytest.fixture
def true_scaffold():
    return ScaffoldAlignments.from_entry_tuples([
        ('scaffold_1', 0, 20, 'contig_1', 0, 20, '+'),
        ('scaffold_1', 20, 100, 'contig_2', 0, 80, '+'),
        ('scaffold_2', 0, 100, 'contig_3', 0, 100, '+')])


@pytest.fixture
def estimated_scaffold():
    return ScaffoldAlignments.from_entry_tuples([
        ('scaffold_1', 0, 20, 'contig_1', 0, 20, '+'),
        ('scaffold_1', 20, 100, 'contig_2', 0, 80, '+'),
        ('scaffold_1', 100, 200, 'contig_3', 0, 100, '+')])


@pytest.fixture
def wrong_scaffold():
    return ScaffoldAlignments.from_entry_tuples([
        ('scaffold_1', 0, 20, 'contig_1', 0, 20, '+'),
        ('scaffold_2', 0, 80, 'contig_2', 0, 80, '+'),
        ('scaffold_3', 0, 100, 'contig_3', 0, 100, '+')])


def test_edge_recall(true_scaffold, estimated_scaffold):
    comparison = ScaffoldComparison(estimated_scaffold, true_scaffold)
    assert comparison.edge_recall() == 1.0


def test_edge_recall(true_scaffold, wrong_scaffold):
    comparison = ScaffoldComparison(wrong_scaffold, true_scaffold)
    assert comparison.edge_recall() == 0


def test_weighted_edge_recall(true_scaffold, estimated_scaffold):
    contig_sizes = {'contig_1': 20, 'contig_2': 80, 'contig_3': 100}
    comparison = ScaffoldComparison(estimated_scaffold, true_scaffold)
    assert comparison.weighted_edge_recall(contig_sizes) == 1.0


def test_weighted_edge_recall_with_wrong_scaffold():
    contig_sizes = {'contig_1': 10, 'contig_2': 10, 'contig_3': 2, 'contig_4': 2}
    true_scaffold = ScaffoldAlignments.from_entry_tuples(
        [('scaffold_1', 0, 20, 'contig_1', 0, 10, '+'),
         ('scaffold_1', 20, 40, 'contig_2', 0, 10, '+'),
         ('scaffold_1', 40, 42, 'contig_3', 0, 2, '+'),
         ('scaffold_1', 42, 44, 'contig_4', 0, 2, '+')]
    )
    wrong_scaffold = ScaffoldAlignments.from_entry_tuples(
        [('scaffold_1', 0, 20, 'contig_1', 0, 10, '+'),
         ('scaffold_1', 20, 40, 'contig_2', 0, 10, '+'),
         ('scaffold_1', 40, 42, 'contig_4', 0, 2, '+'),
         ('scaffold_1', 42, 44, 'contig_3', 0, 2, '+')]
    )
    comparison = ScaffoldComparison(wrong_scaffold, true_scaffold)
    assert comparison.weighted_edge_recall(contig_sizes) == 10 / 18
