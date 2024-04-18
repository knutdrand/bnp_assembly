import itertools
import logging
import time
from typing import List, Dict, Literal
import plotly.express as px
import numpy as np
from bnp_assembly.splitting import split_on_scores
from matplotlib import pyplot as plt
from plotly import express as px
from tqdm import tqdm

from bnp_assembly.contig_graph import DirectedNode, ContigPathSides, ContigPath
from bnp_assembly.graph_objects import Edge
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, total_element_distance, BackgroundMatrix, \
    BackgroundInterMatrices
from bnp_assembly.util import get_all_possible_edges
from .plotting import px as px_func


class PathInteractionMatrix:
    """
    Wrapper around SparseInteractionMatrix that also keeps track
    of the contig path that the matrix represents
    """
    def __init__(self, interaction_matrix: SparseInteractionMatrix, path: List[DirectedNode]):
        self._interaction_matrix = interaction_matrix
        self._path = path

    def matrix_internal_contig_id(self, contig_id):
        # matrix uses numerical ids from 0 to n contigs
        for i, node in enumerate(self._path):
            if node.node_id == contig_id:
                return i

    def matrix_internal_orientation(self, contig_id, orientation) -> Literal["+", "-"]:
        id = self.matrix_internal_contig_id(contig_id)
        return '+' if orientation == self._path[id].orientation else '-'

    def reorder_matrix(self, new_path: List[DirectedNode]):
        """
        Reorders the matrix according to the new path
        """
        relative_path = []
        for node in new_path:
            relative_path.append(DirectedNode(self.matrix_internal_contig_id(node.node_id),
                                              self.matrix_internal_orientation(node.node_id, node.orientation)))
        self._path = new_path
        self._interaction_matrix = self._interaction_matrix.get_matrix_for_path(relative_path, as_raw_matrix=False)

    def flip_node(self, node_id):
        new_path = self._path.copy()
        matrix_contig_id = self.matrix_internal_contig_id(node_id)
        new_path[matrix_contig_id] = new_path[matrix_contig_id].reverse()
        self.reorder_matrix(new_path)
        # flip the correct part of the matrix
        #self._path = new_path
        #self._interaction_matrix.flip_contig(matrix_contig_id)
        #path_relative_to_matrix = [
        #    DirectedNode(self.matrix_internal_contig_id(n.node_id), )
        #]
        #self._interaction_matrix = self._interaction_matrix.get_matrix_for_path(new_path, as_raw_matrix=False)
        #return PathInteractionMatrix(self._interaction_matrix, new_path)

    @classmethod
    def from_interaction_matrix_and_path(cls, matrix: SparseInteractionMatrix, path: List[DirectedNode]):
        """
        matrix is a SparseInteractionMatrix where all contigs are ordered from 0 to n contigs
        This method fetches a reordered matrix according to the path
        """
        new_matrix = matrix.get_matrix_for_path(path, as_raw_matrix=False)
        return cls(new_matrix, path)

    @property
    def raw_matrix(self):
        return self._interaction_matrix.sparse_matrix

    @property
    def path(self):
        return self._path

    def get_contig_left_interaction_matrix(self, contig_id):
        internal_id = self.matrix_internal_contig_id(contig_id)
        start_bin = self._interaction_matrix.global_offset.get_bin_start(internal_id)
        end_bin = self._interaction_matrix.global_offset.get_bin_end(internal_id)
        return self._interaction_matrix.sparse_matrix[0:start_bin, start_bin:end_bin]

    def get_contig_count_and_weights(self, contig_id, direction: Literal["left", "right"]):
        """
        Returns the total distance of all reads from read pos to contig end for reads
        inside the contig that have a pair left outside the contig
        Also returns the weights of those reads (i.e. how many reads there are)
        """
        internal_id = self.matrix_internal_contig_id(contig_id)
        start_bin = self._interaction_matrix._global_offset.contig_first_bin(internal_id)
        end_bin = self._interaction_matrix._global_offset.contig_last_bin(internal_id, inclusive=False)

        if direction == "left":
            submatrix = self._interaction_matrix.sparse_matrix[0:start_bin, start_bin:end_bin]
            pos_outside, pos_inside_contig = np.nonzero(submatrix)
            distance_to_end = pos_inside_contig
        else:
            submatrix = self._interaction_matrix.sparse_matrix[end_bin:, start_bin:end_bin]
            pos_outside, pos_inside_contig = np.nonzero(submatrix)
            distance_to_end = submatrix.shape[1]-pos_inside_contig-1  # distance to end of contig

        if len(pos_outside) == 0:
            return 0, 0

        weights = np.array(submatrix[pos_outside, pos_inside_contig]).ravel()
        total = np.sum(distance_to_end*weights)
        assert total >= 0
        assert np.sum(weights) >= 0
        if total >= 0:
            assert np.sum(weights) >= 0
        return total, np.sum(weights)

    def contig_n_bins(self, contig_id):
        internal_id = self.matrix_internal_contig_id(contig_id)
        return self._interaction_matrix._global_offset._contig_n_bins[internal_id]


class PathTotalReadDistances:
    """
    Represents the total read distance between pairs of reads in a path
    and enables updating this number efficiently when flipping a contig.
    Does this by storing the necessary information for each contig
    on which reads have pairs outside the contig left and right and the weight of these

    Important: Contig-sizes must be on the same scale that is used to calculate distances, i.e.
    number of bins in the contig if binning is used in the interaction matrix
    """
    def __init__(self, initial_value,
                 contig_sizes: Dict[int, int],
                 left_totals: Dict[int, float],
                 left_weights: Dict[int, int],
                 right_totals: Dict[int, float],
                 right_weights: Dict[int, int]):
        self._total_value = initial_value
        self._contig_sizes = contig_sizes
        self._left_totals = left_totals
        self._left_weights = left_weights
        self._right_totals = right_totals
        self._right_weights = right_weights

    def flip_contig(self, contig_id):
        size = self._contig_sizes[contig_id]-1
        new_left_total = size * self._left_weights[contig_id] - self._left_totals[contig_id]
        new_right_total = size * self._right_weights[contig_id] - self._right_totals[contig_id]

        assert new_left_total >= 0, f"New sum of distances is below zero. Size: {size}, weights: {self._left_weights[contig_id]}, old total: {self._left_totals[contig_id]}, new total: {new_left_total}"
        assert new_right_total >= 0, f"New sum of distances is below zero. Size: {size}, weights: {self._right_weights[contig_id]}, old total: {self._right_totals[contig_id]}, new total: {new_right_total}"

        logging.info("\nContig %s, size: %d" % (contig_id, size))
        logging.info(f"Left:  {self._left_totals[contig_id]}/{new_left_total}")
        logging.info(f"Right: {self._right_totals[contig_id]}/{new_right_total}")

        self._total_value -= self._left_totals[contig_id] + self._right_totals[contig_id]
        self._total_value += new_left_total + new_right_total

        self._left_totals[contig_id] = new_left_total
        self._right_totals[contig_id] = new_right_total
        # swap left and right weights
        #self._left_weights[contig_id], self._right_weights[contig_id] = (
        #    self._right_weights[contig_id], self._left_weights[contig_id])

    @classmethod
    def from_interaction_matrix(cls, path: List[DirectedNode], matrix: SparseInteractionMatrix) -> "PathTotalReadDistances":
        path_matrix = PathInteractionMatrix.from_interaction_matrix_and_path(matrix, path)
        value = total_element_distance(path_matrix.raw_matrix)
        left_totals = {}
        left_weights = {}
        right_totals = {}
        right_weights = {}
        contig_sizes = {}

        for contig in tqdm(path):
            left_total, left_weight = path_matrix.get_contig_count_and_weights(contig.node_id, "left")
            right_total, right_weight = path_matrix.get_contig_count_and_weights(contig.node_id, "right")
            left_totals[contig.node_id] = left_total
            left_weights[contig.node_id] = left_weight
            right_totals[contig.node_id] = right_total
            right_weights[contig.node_id] = right_weight
            contig_sizes[contig.node_id] = path_matrix.contig_n_bins(contig.node_id)

        return cls(value, contig_sizes, left_totals, left_weights, right_totals, right_weights)

    @property
    def total_value(self):
        return self._total_value


class PathOptimizer:
    def __init__(self, interaction_matrix: SparseInteractionMatrix, evaluation_function: callable):
        self._interaction_matrix = interaction_matrix
        self._evaluation_function = evaluation_function
        self._current_path_matrix = None

    def init(self, initial_path: List[DirectedNode]):
        self._current_path_matrix = PathInteractionMatrix.from_interaction_matrix_and_path(
            self._interaction_matrix, initial_path)

    def evaluate_path(self) -> float:
        return self._evaluation_function(self._current_path_matrix.raw_matrix)

    def run(self) -> List[DirectedNode]:

        # try flipping one and one node
        # if the score improves, keep the change
        current_score = self.evaluate_path()

        for i, node in enumerate(self._current_path_matrix.path):
            logging.info(node)
            #logging.info(f"\nContig {node}. Current score is {current_score}")
            t0 = time.perf_counter()
            self._current_path_matrix.flip_node(node.node_id)
            new_score = self.evaluate_path()
            if new_score < current_score:
                logging.info(f"Improved score from {current_score} to {new_score} by flipping node {node}")
                current_score = new_score
            else:
                self._current_path_matrix.flip_node(node.node_id)

        return self._current_path_matrix.path


class TotalDistancePathOptimizer:
    def __init__(self, initial_path: List[DirectedNode], interaction_matrix: SparseInteractionMatrix):
        logging.info("Initing")
        self._current_path = initial_path.copy()
        self._distances = PathTotalReadDistances.from_interaction_matrix(initial_path, interaction_matrix)
        logging.info("Inited")

    def evaluate_path(self) -> float:
        return self._distances.total_value

    def run(self) -> List[DirectedNode]:
        # try flipping one and one node
        # if the score improves, keep the change
        current_score = self.evaluate_path()

        for i, node in enumerate(self._current_path):
            logging.info(f"Current score is {current_score}")
            t0 = time.perf_counter()
            self._distances.flip_contig(node.node_id)
            new_score = self.evaluate_path()
            assert new_score > 0
            if new_score < current_score:
                current_score = new_score
                logging.info(f"Improved score to {new_score} by flipping node {node}")
                self._current_path[i] = node.reverse()
            else:
                logging.info(f"Flipping node {node} did not improve score, new score is {new_score}")
                # flip back
                self._distances.flip_contig(node.node_id)

        return self._current_path


def flip_contigs_in_splitted_path(interaction_matrix: SparseInteractionMatrix, paths: List[ContigPathSides]) \
        -> List[ContigPathSides]:
    """
    Goes through each path and optimizes flippings
    """
    new_paths = []
    for path in paths:
        directed_nodes = path.directed_nodes
        optimizer = TotalDistancePathOptimizer(directed_nodes, interaction_matrix)
        new_directed_nodes = optimizer.run()
        logging.info(f"Optimized path:\nOld: {directed_nodes}\nNew: {new_directed_nodes}")
        path = ContigPath.from_directed_nodes(new_directed_nodes)
        new_paths.append(path)

    return new_paths

def flip_contigs_in_splitted_path_path_optimizer(interaction_matrix: SparseInteractionMatrix, paths: List[ContigPathSides], evaluation_function) \
        -> List[ContigPathSides]:
    """
    Goes through each path and optimizes flippings
    """
    new_paths = []
    for path in paths:
        directed_nodes = path.directed_nodes
        optimizer = PathOptimizer(interaction_matrix, evaluation_function)
        optimizer.init(directed_nodes)
        optimizer._interaction_matrix.plot_submatrix(0, len(directed_nodes)-1)
        new_directed_nodes = optimizer.run()
        logging.info(f"Optimized path:\nOld: {directed_nodes}\nNew: {new_directed_nodes}")
        path = ContigPath.from_directed_nodes(new_directed_nodes)
        optimizer._current_path_matrix._interaction_matrix.plot_submatrix(0, len(directed_nodes)-1)
        plt.show()
        new_paths.append(path)

    return new_paths


def optimize_splitted_path(paths: List[ContigPathSides], interaction_matrix,
                           distances_and_weights, distance_func) -> List[ContigPathSides]:
    new_paths = []
    for path in paths:
        logging.info(f"Opimizing subpath: {path}")
        directed_nodes = path.directed_nodes
        path_contig_sizes = np.array([interaction_matrix.contig_n_bins[contig.node_id] for contig in directed_nodes])
        scorer = LogProbSumOfReadDistancesDynamicScores(directed_nodes.copy(), path_contig_sizes,
                                                    distances_and_weights, distance_func=distance_func)
        if len(directed_nodes) > 5:
            scorer.optimize_positions()
            scorer.optimize_flippings()
        else:
            scorer.try_all_possible_paths()

        new_path = ContigPath.from_directed_nodes(scorer._path)
        new_paths.append(new_path)
        logging.info(f"New path:         {new_path}")
    return new_paths


class InteractionDistancesAndWeights:
    """
    Represents the distances and weights of all bins for a given edge
    (as if the two contigs of the edge are next to each other).
    This can then be used to compute
    stuff when a known distance between the contigs is known
    """
    def __init__(self, distances: Dict[Edge, float], weights: Dict[Edge, float]):
        self.distances = distances
        self.weights = weights

    def score_path(self, path: List[DirectedNode]):
        pass

    @classmethod
    def from_sparse_interaction_matrix(cls, sparse_matrix: SparseInteractionMatrix):
        n_contigs = sparse_matrix.n_contigs
        distances = {}
        weights = {}
        for edge in tqdm(get_all_possible_edges(n_contigs), total=(n_contigs*2)**2):
            submatrix = sparse_matrix.get_edge_interaction_matrix(edge, orient_according_to_nearest_interaction=True)
            rows, cols = np.nonzero(submatrix)
            weights[edge] = np.array(submatrix[rows, cols]).ravel()
            # distance between reads when the contigs are together is the
            # sum of rows (distance from edge inside contig1) and cols (distance from edge inside contig2)
            distances[edge] = np.abs(rows+cols)
            assert np.all(distances[edge] >= 0)
            assert np.all(distances[edge] < submatrix.shape[0]+submatrix.shape[1])

        return cls(distances, weights)


class LogProbSumOfReadDistancesDynamicScores:
    """
    Represents a path of contigs and data that can be used to efficiently
    recompute the sum of log of probs of read distances when contigs are flipped or moved
    """
    def __init__(self, initial_path: List[DirectedNode], contig_sizes: np.ndarray,
                 distances_and_weights: InteractionDistancesAndWeights,
                 distance_func: callable = np.log):
        self._distance_func = distance_func
        self._path = initial_path
        self._contig_sizes = contig_sizes
        # better to represent as a dict, since path is changing
        self._contig_sizes_dict = {node.node_id: size for node, size in zip(initial_path, contig_sizes)}
        self._contig_sizes = None
        self.n_contigs = len(contig_sizes)
        self._distances_and_weights = distances_and_weights
        self._score_matrix = np.zeros((self.n_contigs, self.n_contigs))
        self._edge_score_cache = {}
        self._initialize_score_matrix()
        self._current_score = None

    def contig_index_in_path(self, contig):
        for i, node in enumerate(self._path):
            if node.node_id == contig:
                return i
        raise Exception(f"Contig {contig} not found in path {self._path}")

    def contig_position_in_path(self, contig):
        """Gives offset of node"""
        offset = sum(self._contig_sizes_dict[node.node_id] for node in self._path[:self.contig_index_in_path(contig)])
        return offset
        #offset = np.insert(np.cumsum(self._contig_sizes), 0, 0)
        #return offset[self.contig_index_in_path(contig)]

    def distance_between_contigs_in_path(self, contig1: int, contig2: int):
        offset1 = self.contig_position_in_path(contig1)
        offset2 = self.contig_position_in_path(contig2)
        if offset1 < offset2:
            dist = offset2 - offset1 - self._contig_sizes_dict[contig1] + 1
        else:
            dist = offset1 - offset2 - self._contig_sizes_dict[contig2] + 1

        assert dist <= sum(self._contig_sizes_dict.values()) + self._contig_sizes_dict[contig1] + self._contig_sizes_dict[contig2]
        return dist

    def compute_edge_score(self, i, j):
        """
        Computes and sets score between contig at position i and j in path
        """
        contig1, contig2 = self._path[i], self._path[j]
        edge = Edge(contig1.right_side, contig2.left_side)
        distance_between_contigs = self.distance_between_contigs_in_path(contig1.node_id, contig2.node_id)

        cache_id = (edge, distance_between_contigs)
        if cache_id in self._edge_score_cache:
            score = self._edge_score_cache[cache_id]
        else:
            distances = self._distances_and_weights.distances[edge] + distance_between_contigs
            weights = self._distances_and_weights.weights[edge]
            score = np.sum(self._distance_func(distances) * weights)
            self._edge_score_cache[cache_id] = score

        self._score_matrix[i, j] = score
        self._score_matrix[j, i] = score
        return score

    def _initialize_score_matrix(self):
        """
        Fills a score matrix for the given path. The sum of the matrix will be the score
        """
        t0 = time.perf_counter()
        self._score_matrix = np.zeros((self.n_contigs, self.n_contigs))
        for i in range(self.n_contigs):
            for j in range(self.n_contigs):
                if j > i:
                    self.compute_edge_score(i, j)
        #logging.info(f"Initialized score matrix in {time.perf_counter()-t0} seconds")

    def update_scores_affected_by_contig(self, contig):
        index = self.contig_index_in_path(contig)
        # recompute all scores that involves this contig
        for other_contig in range(self.n_contigs):
            if other_contig != index:
                first = min(index, other_contig)
                second = max(index, other_contig)
                self.compute_edge_score(first, second)

    def flip_contig(self, contig):
        index = self.contig_index_in_path(contig)
        self._path[index] = self._path[index].reverse()
        # recompute all scores that involves this contig
        self.update_scores_affected_by_contig(contig)

    def move_contig_right(self, contig):
        index = self.contig_index_in_path(contig)
        if index == self.n_contigs-1:
            return
        self._path[index], self._path[index+1] = self._path[index+1], self._path[index]
        #self._initialize_score_matrix()
        #self.update_scores_affected_by_contig(self._path[index-1].node_id)
        self.update_scores_affected_by_contig(self._path[index].node_id)
        self.update_scores_affected_by_contig(self._path[index+1].node_id)

    def move_contig_left(self, contig):
        index = self.contig_index_in_path(contig)
        if index == 0:
            return
        self._path[index], self._path[index -1] = self._path[index - 1], self._path[index]
        self.update_scores_affected_by_contig(self._path[index].node_id)
        self.update_scores_affected_by_contig(self._path[index - 1].node_id)

    def score(self):
        return np.sum(self._score_matrix)

    def optimize_flippings(self):
        current_score = self.score()
        for i, node in enumerate(self._path):
            self.flip_contig(node.node_id)
            new_score = self.score()
            if new_score < current_score:
                #logging.info(f"Improved score from {current_score} to {new_score} by flipping node {node}")
                current_score = new_score
            else:
                #logging.info(f"Flipping node {node} did not improve score, new score is {new_score}, best score is {current_score}")
                self.flip_contig(node.node_id)

        return self._path

    def find_best_position_for_contig(self, contig):
        """Finds best position and moves the contig to that position"""
        # start at beginning of matrix and try all positions
        original_path = self._path.copy()
        index = self.contig_index_in_path(contig)
        del self._path[index]
        self._path.insert(0, original_path[index])
        self._initialize_score_matrix()

        best_position = 0
        best_score = self.score()
        all_scores = []
        for i in range(1, self.n_contigs):
            self.move_contig_right(contig)
            new_score = self.score()
            if new_score < best_score:
                best_score = new_score
                best_position = i

            all_scores.append(new_score)

        #logging.info(f"Best position for contig {contig} is {best_position} with score {best_score}. Original pos was {index}")
        #px.line(x=np.arange(len(all_scores)), y=all_scores, title=f"Contig {contig}").show()
        # remove from end where contig is now and insert at pos
        contig = self._path.pop()
        self._path.insert(best_position, contig)
        # reinitialze
        self._initialize_score_matrix()
        #logging.info(f"Score after reinitializing is {self.score()}")

    def optimize_positions(self):
        current_score = self.score()
        logging.info(f"Score before optimizing positions: {current_score}")
        for i, node in enumerate(self._path.copy()):
            self.find_best_position_for_contig(node.node_id)
        logging.info(f"Score after optimizing positions: {self.score()}")
        return self._path

    def optimize_by_moving_subpaths(self, raw_interaction_matrix: SparseInteractionMatrix):
        # Splits, moves parts between splitting
        # Idea is to split with strict threshold, keeping what is likely meant to be together together
        # and moving these things around to try to reach a better score
        t0 = time.perf_counter()
        path = ContigPath.from_directed_nodes(self._path)
        path_matrix = raw_interaction_matrix.get_matrix_for_path(path.directed_nodes, as_raw_matrix=False)
        inter_distribution = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix)
        #splitted_paths = split_using_interaction_matrix(path, path_matrix, threshold=0.5)
        splitted_paths = split_using_inter_distribution(path_matrix, inter_distribution, path, threshold=0.0005)
        splitted_paths_lengths = [len(subpath.directed_nodes) for subpath in splitted_paths]
        indexes_with_splitting = np.insert(np.cumsum(splitted_paths_lengths), 0, 0)
        logging.info(f"Splitted paths lengths: {splitted_paths_lengths}")
        indexes_next_to_multicontig_subpaths = []
        prev_length = 0
        for i, length in enumerate(splitted_paths_lengths):
            if length > 1 or prev_length > 1:
                indexes_next_to_multicontig_subpaths.append(indexes_with_splitting[i])
            prev_length = length
        logging.info(f"Will move multi-contig paths to {indexes_next_to_multicontig_subpaths}")

        for subpath in splitted_paths:
            directed_nodes = subpath.directed_nodes
            #logging.info(f"Trying to move subpath {directed_nodes}")
            if len(directed_nodes) == 1:
                # single contig, no point in trying to move
                continue
            # try all possible positions for the subpath
            best_path = self._path.copy()
            best_score = self.score()
            # remove from the original path
            for node in directed_nodes:
                self._path.remove(node)
            path_without_nodes = self._path.copy()
            #for i in range(len(self._path)):
            for i in indexes_next_to_multicontig_subpaths:
                for dir in range(2):
                    # reset
                    self._path = path_without_nodes
                    if dir == 0:
                        self._path = self._path[:i] + directed_nodes + self._path[i:]
                    else:
                        self._path = self._path[:i] + [n.reverse() for n in directed_nodes[::-1]] + self._path[i:]
                    self._initialize_score_matrix()
                    new_score = self.score()
                    if new_score < best_score:
                        best_score = new_score
                        best_path = self._path.copy()
                        #logging.info(f"   Found new path with score {best_score}: {best_path} by moving {directed_nodes} to position {i}")

            self._path = best_path
            self._initialize_score_matrix()
        logging.info(f"Optimized by moving subpaths in {time.perf_counter()-t0} seconds")
        return self._path

    def try_all_possible_paths(self):
        """
        Tries all possible paths and returns the best one,
        works when there are few contigs
        """
        n = len(self._path)
        possible_nodes = [node.node_id for node in self._path]
        possible_orientations = ['+', '-']
        best_path = self._path
        best_score = self.score()
        all_possible_orientations = list(itertools.product(possible_orientations, repeat=n))
        #all_possible_orientations = [["+" for _ in range(n)]]
        for nodes in itertools.permutations(possible_nodes, len(possible_nodes)):
            for orientations in all_possible_orientations:
                self._path = [DirectedNode(node, orientation) for node, orientation in zip(nodes, orientations)]
                self._initialize_score_matrix()
                new_score = self.score()
                if new_score < best_score:
                    best_score = new_score
                    best_path = self._path

        logging.info(f"Found new path with score {best_score}: {best_path}")
        self._path = best_path
        self._initialize_score_matrix()

    def move_contig_to_position(self, contig, position):
        new_path = self._path.copy()
        current_pos = self.contig_index_in_path(contig)
        new_path[current_pos] = "delete"
        new_path.insert(position, self._path[self.contig_index_in_path(contig)])
        del new_path[new_path.index("delete")]
        self._path = new_path
        self._initialize_score_matrix()

    def get_path(self):
        return self._path


def split_using_interaction_matrix(path, path_matrix, threshold=0.1):
    background = BackgroundMatrix.from_sparse_interaction_matrix(path_matrix)
    minimum_assumed_chromosome_size_in_bins = path_matrix.sparse_matrix.shape[1] // 300
    logging.info("Minimum assumed chromosome size in bins: %d" % minimum_assumed_chromosome_size_in_bins)
    edge_scores = {
        edge: path_matrix.edge_score(i + 1, minimum_assumed_chromosome_size_in_bins, background_matrix=background) for
        i, edge in enumerate(path.edges)}
    # todo, save plot to logs instead:
    #px.bar(x=[str(e) for e in edge_scores.keys()], y=list(edge_scores.values())).show()
    # split this path based on the scores from distance matrix
    logging.info(f"Path before splitting {path.directed_nodes}")
    splitted_paths = split_on_scores(path, edge_scores, threshold=threshold, keep_over=True)
    return splitted_paths


def get_splitting_edge_scores(path_matrix, background_inter_matrices: BackgroundInterMatrices,
                              path: ContigPathSides,
                              threshold=0.05):

    minimum_assumed_chromosome_size_in_bins = path_matrix.sparse_matrix.shape[1] // 50

    edge_scores = {}
    max_size = background_inter_matrices.matrices.shape[1]
    logging.info(f"Max size when splitting: {max_size}")
    logging.info(f"Minimum assumed chromosome size: {minimum_assumed_chromosome_size_in_bins}")
    for i, edge in enumerate(path.edges):
        matrix1, matrix2 = path_matrix.edge_score(i + 1, minimum_assumed_chromosome_size_in_bins, return_matrices=True)
        # matrices have direction directly from the main matrix, i.e. not
        # oriented according to edge, nearest interaction is bottom left
        matrix1 = matrix1[-max_size:, :max_size]
        matrix2 = matrix2[-max_size:, :max_size]
        percentile1 = background_inter_matrices.get_percentile2(matrix1.shape[0], matrix1.shape[1], matrix1.sum())
        percentile2 = background_inter_matrices.get_percentile2(matrix2.shape[0], matrix2.shape[1], matrix2.sum())
        # percentile is ratio of background with larger sum, low percentile indicates high counts and one should not split
        # we care about the lowest percentile when looking both directions
        edge_scores[edge] = min(percentile1, percentile2)
        #print(f"Edge {edge} has score {edge_scores[edge]}. Percentile1: {percentile1}, percentile2: {percentile2}.")
        #print(f"  Shape/sum1: {matrix1.shape}, {matrix1.sum()}")
        #print(f"  Shape/sum2: {matrix2.shape}, {matrix2.sum()}")

    return edge_scores


def split_using_inter_distribution(path_matrix, background_inter_matrices: BackgroundInterMatrices, path: ContigPathSides,
                                   threshold=0.05):
    edge_scores = get_splitting_edge_scores(path_matrix, background_inter_matrices, path, threshold=threshold)
    #px_func(name="main").bar(x=[str(e) for e in edge_scores.keys()], y=list(edge_scores.values()), title="Splitting scores")
    logging.info(f"Path before splitting {path.directed_nodes}")
    splitted_paths = split_on_scores(path, edge_scores, threshold=threshold, keep_over=False)
    return splitted_paths
