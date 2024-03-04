import logging

from matplotlib import pyplot as plt
from shared_memory_wrapper import from_file

from bnp_assembly.graph_objects import Edge, NodeSide

logging.basicConfig(level=logging.INFO)
import numpy as np
import scipy
import pytest
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.contig_path_optimization import PathOptimizer, PathTotalReadDistances, TotalDistancePathOptimizer, \
    InteractionDistancesAndWeights, LogProbSumOfReadDistancesDynamicScores
from bnp_assembly.sparse_interaction_matrix import average_element_distance, BinnedNumericGlobalOffset, \
    SparseInteractionMatrix, total_element_distance


@pytest.mark.xfail
def test_acceptance():
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 2, 1: 2, 2: 2}, 1)
    matrix = SparseInteractionMatrix.empty(g)

    data = np.arange(36).reshape(6, 6)
    data[0, 5] = 5
    matrix.set_matrix(scipy.sparse.lil_matrix(data).tocsr())

    print(matrix.nonsparse_matrix)

    optimizer = PathOptimizer(matrix, total_element_distance)
    initial_path = [DirectedNode(0, '+'), DirectedNode(1, '+'), DirectedNode(2, '+')]
    optimizer.init(initial_path)
    new_path = optimizer.run()

    assert new_path == [DirectedNode(0, '+'), DirectedNode(1, '+'), DirectedNode(2, '-')]

    print(new_path)


@pytest.mark.xfail
def test_path_total_distances():
    d = PathTotalReadDistances(
        initial_value=50,
        contig_sizes={0: 6},
        left_totals={0: 13},
        left_weights={0: 6},
        right_totals={0: 4},
        right_weights={0: 2},
    )

    print(d.total_value)

    d.flip_contig(0)
    correct = 50 - 4 + 4 * 2 - 13 + 5*2 + 4*3 + 1*1
    #correct = 50 - 13 - 4 + 6 * 6 + 6 * 2 - 13 - 4
    assert d.total_value == correct

    d.flip_contig(0)
    assert d.total_value == 50


def test_total_read_optimizer_acceptance():
    #g = BinnedNumericGlobalOffset.from_contig_sizes({0: 2, 1: 2, 2: 2}, 1)
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 2, 1: 2}, 1)
    matrix = SparseInteractionMatrix.empty(g)

    data = [
        [0, 1, 0, 2],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0]
    ]
    matrix.set_matrix(scipy.sparse.lil_matrix(data).tocsr())
    assert total_element_distance(matrix.nonsparse_matrix) == 1*1 + 2*3

    initial_path = [DirectedNode(0, '+'), DirectedNode(1, '+')]
    optimizer = TotalDistancePathOptimizer(initial_path, matrix)
    assert optimizer.evaluate_path() == 7
    optimizer._distances.flip_contig(0)
    assert optimizer.evaluate_path() == 1*1 + 2*2
    optimizer._distances.flip_contig(0)
    assert optimizer.evaluate_path() == 7
    optimizer.run()

    # reverse initial path should not change anything
    initial_path = [DirectedNode(1, '-'), DirectedNode(0, '-')]
    optimizer = TotalDistancePathOptimizer(initial_path, matrix)
    assert optimizer.evaluate_path() == 7

    initial_path = [DirectedNode(0, '-'), DirectedNode(1, '+')]
    optimizer = TotalDistancePathOptimizer(initial_path, matrix)
    assert optimizer.evaluate_path() == 5


def test_total_read_optimizer_random_matrices():
    """Checks that values computed by total distance path optimizer matches those computes
    from actual matrix"""
    np.random.seed(0)
    contig_sizes = {0: 3, 1: 5, 2: 10, 3: 2}
    total_size = sum(contig_sizes.values())
    matrix = np.random.randint(0, 10, (total_size, total_size))
    matrix = np.maximum(matrix, matrix.T)  # make symmetric
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes(contig_sizes, 1)
    interaction_matrix = SparseInteractionMatrix.empty(global_offset)
    interaction_matrix.set_matrix(scipy.sparse.lil_matrix(matrix).tocsr())

    initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]

    assert np.all(interaction_matrix.get_matrix_for_path(initial_path).toarray() == matrix)

    optimizer = TotalDistancePathOptimizer(initial_path, interaction_matrix)
    assert optimizer.evaluate_path() == total_element_distance(interaction_matrix.sparse_matrix)

    for contig in range(len(contig_sizes)):
        optimizer._distances.flip_contig(contig)
        new_path = initial_path.copy()
        new_path[contig] = new_path[contig].reverse()
        matrix_with_flipped_contig = interaction_matrix.get_matrix_for_path(new_path, as_raw_matrix=True)
        assert optimizer.evaluate_path() == total_element_distance(matrix_with_flipped_contig)

        # flip back
        optimizer._distances.flip_contig(contig)


def test_total_element_distance():
    matrix = np.array([
       [1, 2, 0, 2, 2],
       [2, 0, 2, 1, 1],
       [0, 2, 0, 1, 1],
       [2, 1, 1, 1, 1],
       [2, 1, 1, 1, 1]])

    assert total_element_distance(matrix) == 27


def test_total_read_optimizer_large_interaction_matrix():
    np.random.seed(0)
    n_contigs = 10
    contig_sizes = {i: 50 for i in range(n_contigs)}
    total_size = sum(contig_sizes.values())
    matrix = np.zeros((total_size, total_size))
    for i in range(total_size):
        for j in range(total_size):
            dist = abs(i-j)
            prob = 0.5 * ((total_size - dist) / total_size)**40
            if np.random.random() < prob:
                matrix[i, j] = 10 * (total_size - abs(i-j)) / total_size

    matrix = np.maximum(matrix, matrix.T)  # make symmetric
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes(contig_sizes, 1)
    interaction_matrix = SparseInteractionMatrix.empty(global_offset)
    interaction_matrix.set_matrix(scipy.sparse.lil_matrix(matrix).tocsr())
    #fig, ax = interaction_matrix.plot_submatrix(0, 2)
    #plt.show()

    initial_path = [DirectedNode(contig, '+') for contig in range(len(contig_sizes))]
    optimizer = TotalDistancePathOptimizer(initial_path, interaction_matrix)
    new_path = optimizer.run()
    assert new_path == initial_path

    for contig in range(len(contig_sizes)):
        optimizer._distances.flip_contig(contig)
        new_path = initial_path.copy()
        new_path[contig] = new_path[contig].reverse()
        matrix_with_flipped_contig = interaction_matrix.get_matrix_for_path(new_path, as_raw_matrix=True)

        truth = optimizer.evaluate_path()
        measured = total_element_distance(matrix_with_flipped_contig)
        assert abs(truth - measured) < 2, (truth, measured)

        # flip back
        optimizer._distances.flip_contig(contig)


def test_logprob_dynamic_scores():
    interaction_matrix = np.array([
        [1, 0, 0, 1, 2],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [2, 0, 1, 0, 1]
    ])
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes({0: 2, 1: 2, 2: 1}, 1)
    interaction_matrix = SparseInteractionMatrix.from_np_matrix(global_offset, interaction_matrix)
    #submatrix = interaction_matrix.get_edge_interaction_matrix(Edge(NodeSide(0, 'r'), NodeSide(1, 'l')), orient_according_to_nearest_interaction=True)
    #print(submatrix.toarray())

    dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(interaction_matrix)
    path = [DirectedNode(contig, '+') for contig in range(3)]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)

    assert scorer.contig_position_in_path(0) == 0
    assert scorer.contig_position_in_path(2) == 4
    assert scorer.distance_between_contigs_in_path(0, 1) == 1

    score = scorer.score()
    print(scorer._score_matrix)
    assert score == 2 * (1 * 3 + 2 * 4 + 1 * 2)

    path = [DirectedNode(2, '-'), DirectedNode(1, '-'), DirectedNode(0, '-')]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)
    score = scorer.score()
    assert score == 26

    path = [DirectedNode(0, '+'), DirectedNode(1, '-'), DirectedNode(2, '+')]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)
    score = scorer.score()
    assert score == 11*2

    # test flipping
    path = [DirectedNode(contig, '+') for contig in range(3)]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)
    scorer.flip_contig(1)
    print(scorer._score_matrix)
    assert scorer.score() == 22

    scorer.flip_contig(1)
    print(scorer._score_matrix)
    assert scorer.score() == 26

    path = [DirectedNode(contig, '+') for contig in range(3)]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)
    scorer.move_contig_right(0)
    assert scorer.score() == 2*9
    scorer.move_contig_right(0)
    scorer.move_contig_left(0)
    assert scorer.score() == 2*9
    scorer.move_contig_left(0)
    assert scorer.score() == 26

    path = [DirectedNode(contig, '+') for contig in range(3)]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)
    scorer.find_best_position_for_contig(2)


def test_move_contig_right():
    n_contigs = 10
    interaction_matrix = np.random.randint(0, 4, (2*n_contigs, 2*n_contigs))
    interaction_matrix = np.maximum(interaction_matrix, interaction_matrix.T)
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes({i: 2 for i in range(n_contigs)}, 1)
    interaction_matrix = SparseInteractionMatrix.from_np_matrix(global_offset, interaction_matrix)
    dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(interaction_matrix)
    path = [DirectedNode(contig, '+') for contig in range(n_contigs)]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, interaction_matrix.contig_n_bins, dists_weights, np.abs)

    for i in range(2):
        scorer.move_contig_right(0)
        score = scorer.score()
        scorer._initialize_score_matrix()
        assert scorer.score() == score


def test_bug():
    global_offset = BinnedNumericGlobalOffset.from_contig_sizes({0: 2, 1: 1, 2: 2}, 1)
    matrix = np.zeros((5, 5)) + 1
    for i in range(5):
        for j in range(5):
            matrix[i, j] = 4-abs(i-j)

    matrix[1, 3] = 10
    matrix[3, 1] = 10

    print(matrix)
    matrix = SparseInteractionMatrix.from_np_matrix(global_offset, matrix)
    dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(matrix)
    path = [DirectedNode(contig, '+') for contig in range(3)]
    scorer = LogProbSumOfReadDistancesDynamicScores(path, matrix.contig_n_bins, dists_weights, np.abs)
    score = scorer.score()
    score_matrix = scorer._score_matrix
    print(score_matrix)

    matrix.plot_submatrix(0, 2)
    scorer.move_contig_right(0)
    scorer.move_contig_right(0)
    print(score_matrix)

    matrix2 = matrix.get_matrix_for_path(scorer._path, as_raw_matrix=True)
    print(scorer._path)
    print(matrix2.toarray())


def test_integration_real_case():
    matrix = from_file("interaction_matrix_test.npz")
    distance_pmf = np.load("distance_pmf.npy")
    n_contigs = matrix.n_contigs
    matrix.plot_submatrix(0, matrix.n_contigs - 1)

    testpath = [DirectedNode(contig, '+') for contig in range(matrix.n_contigs)]
    distance_func = lambda dist: -distance_pmf[dist]
    dists_weights = InteractionDistancesAndWeights.from_sparse_interaction_matrix(matrix)

    path_contig_sizes = np.array([matrix.contig_n_bins[contig.node_id] for contig in testpath])
    scorer = LogProbSumOfReadDistancesDynamicScores(testpath.copy(), path_contig_sizes, dists_weights,
                                                    distance_func=distance_func)

    scorer.find_best_position_for_contig(0)
    new_path = scorer._path
    new_matrix = matrix.get_matrix_for_path(new_path, as_raw_matrix=False)
    new_matrix.plot_submatrix(0, n_contigs - 1)
    #plt.show()

    logging.info(f"Old path: {testpath}")
    logging.info(f"New path: {new_path}")

    assert new_path == testpath


if __name__ == "__main__":
    #test_acceptance()
    #test_total_read_optimizer_acceptance()
    #test_total_read_optimizer_large_interaction_matrix()
    #test_logprob_dynamic_scores()
    test_move_contig_right()
    print("done")


