import logging
import time
from typing import List, Dict, Literal
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from bnp_assembly.contig_graph import DirectedNode, ContigPathSides, ContigPath
from bnp_assembly.graph_objects import Edge
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, total_element_distance
from bnp_assembly.util import get_all_possible_edges


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
        intra_distances = self._distances_and_weights.distances[edge]
        contig_sizes = self._contig_sizes_dict
        node1_size = contig_sizes[contig1.node_id]
        node2_size = contig_sizes[contig2.node_id]
        assert np.all(intra_distances < node1_size + node2_size)
        distances = self._distances_and_weights.distances[edge] + distance_between_contigs
        weights = self._distances_and_weights.weights[edge]
        score = np.sum(self._distance_func(distances) * weights)
        self._score_matrix[i, j] = score
        self._score_matrix[j, i] = score
        return score

    def _initialize_score_matrix(self):
        """
        Fills a score matrix for the given path. The sum of the matrix will be the score
        """
        self._score_matrix = np.zeros((self.n_contigs, self.n_contigs))
        for i in range(self.n_contigs):
            for j in range(self.n_contigs):
                if j > i:
                    self.compute_edge_score(i, j)

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
            logging.info(node)
            self.flip_contig(node.node_id)
            new_score = self.score()
            if new_score < current_score:
                logging.info(f"Improved score from {current_score} to {new_score} by flipping node {node}")
                current_score = new_score
            else:
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

        logging.info(f"Best position for contig {contig} is {best_position} with score {best_score}. Original pos was {index}")
        #px.line(x=np.arange(len(all_scores)), y=all_scores, title=f"Contig {contig}").show()
        # remove from end where contig is now and insert at pos
        contig = self._path.pop()
        self._path.insert(best_position, contig)
        # reinitialze
        self._initialize_score_matrix()
        logging.info(f"Score after reinitializing is {self.score()}")

    def optimize_positions(self):
        current_score = self.score()
        logging.info(f"Score before optimizing positions: {current_score}")
        for i, node in enumerate(self._path.copy()):
            self.find_best_position_for_contig(node.node_id)
        logging.info(f"Score after optimizing positions: {self.score()}")
        return self._path

    def move_contig_to_position(self, contig, position):
        new_path = self._path.copy()
        current_pos = self.contig_index_in_path(contig)
        new_path[current_pos] = "delete"
        new_path.insert(position, self._path[self.contig_index_in_path(contig)])
        del new_path[new_path.index("delete")]
        self._path = new_path
        self._initialize_score_matrix()
