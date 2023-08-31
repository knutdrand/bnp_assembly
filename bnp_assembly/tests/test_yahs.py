import numpy as np
import pytest

from bnp_assembly.scaffold_splitting.binned_bayes import Yahs


@pytest.fixture
def ones_yahs():
    matrix = np.ones((100, 100))
    start_stops = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    start_stop_dict = {i: start_stops[i] for i in range(len(start_stops))}
    return Yahs(matrix, start_stop_dict)


@pytest.fixture
def diag_yahs():
    matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            matrix[i, j] = abs(i - j) + 1
    start_stops = [(0, 5), (5, 10)]
    start_stop_dict = {i: start_stops[i] for i in range(len(start_stops))}
    return Yahs(matrix, start_stop_dict)


@pytest.fixture
def block_yahs():
    matrix = np.array([[2, 2, 0, 0],
                       [2, 2, 0, 0],
                       [0, 0, 2, 2],
                       [0, 0, 2, 2]])
    start_stops = [(0, 2), (2, 4)]
    start_stop_dict = {i: start_stops[i] for i in range(len(start_stops))}
    return Yahs(matrix, start_stop_dict)


def test_yahs_acceptance(ones_yahs):
    assert len(ones_yahs.score_vector()) == 9


def test_filling_rate(ones_yahs):
    for d in range(1, 10):
        for m in range(9):
            assert ones_yahs.f(m, m + 1, d) == 1.0


def test_delta(ones_yahs):
    for n in range(9):
        for i in range(10):
            for j in range(10):
                assert ones_yahs.delta(n, i, j) == abs(i - j)
                assert ones_yahs.delta(i, j) == abs(i - j)
                assert ones_yahs.delta(n, n + 1, i, j) == abs(10 - i + j)


def test_E(ones_yahs):
    for d in range(1, 10):
        assert ones_yahs.E(d) == 1.0


def test_cells(ones_yahs):
    for n in range(10):
        for i in range(10):
            for j in range(10):
                if i != j:
                    assert ones_yahs.c(n, i, j) == 1.0
                    assert ones_yahs.w(n, i, j) == 1.0
                if n < 9:
                    if 10 - i + j < 10:
                        assert ones_yahs.c(n, n + 1, i, j) == 1.0
                        assert ones_yahs.w(n, n + 1, i, j) == 1.0


def test_E_diag(diag_yahs):
    for d in range(1, 5):
        assert diag_yahs.E(d) == d + 1


def test_w_diag(diag_yahs):
    for i in range(10):
        for j in range(max(i - 4, 0), min(i + 4, 10)):
            if i != j:
                assert diag_yahs.w(i, j) == 1.0

    for i in range(5):
        for j in range(i):
            assert diag_yahs.c(0, 1, i, j) == diag_yahs.E(diag_yahs.delta(0, 1, i, j))
            assert diag_yahs.w(0, 1, i, j) == 1.0


def test_diag_fill_rate(diag_yahs):
    for d in range(1, 5):
        for m in range(1):
            assert diag_yahs.f(m, m + 1, d) == 1.0


def test_block_scores(block_yahs):
    assert block_yahs.score_vector() == [0]
