import logging
import random
from typing import Union, List

import numpy as np
from matplotlib import pyplot as plt

from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.graph_objects import NodeSide
from bnp_assembly.sparse_interaction_based_distance import get_bayesian_edge_probs, get_inter_background, \
    get_intra_background
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix
from bnp_assembly.plotting import px

class CompundNode:
    def __init__(self, nodes: List[Union[DirectedNode, 'CompundNode']]):
        self.nodes = nodes

    def reverse(self):
        return CompundNode([node.reverse() for node in self.nodes][::-1])

    def __str__(self):
        return f"[{self.nodes}]"

    def __repr__(self):
        return str(self)

    def flatten(self):
        to_return = []
        for node in self.nodes:
            if isinstance(node, CompundNode):
                to_return.extend(node.flatten())
            else:
                to_return.append(node)
        return to_return


class IterativePathJoiner:
    def __init__(self, interaction_matrix: SparseInteractionMatrix):
        self._interaction_matrix = interaction_matrix
        self._current_path = []
        self._current_distance_matrix = None
        self._inter_background = None
        self._intra_background = None
        self.shuffle() # start with random path
        self._compute_distance_matrix()

    def shuffle(self):
        n_contigs = self._interaction_matrix.n_contigs
        initial_path = [DirectedNode(contig, '+') for contig in range(n_contigs)]
        random.shuffle(initial_path)
        self._current_path = initial_path
        self._interaction_matrix = self._interaction_matrix.get_matrix_for_path(initial_path, as_raw_matrix=False)
        self._get_backgrounds()
        self._compute_distance_matrix()

    def _get_backgrounds(self):
        self._inter_background = get_inter_background(self._interaction_matrix)
        self._intra_background = get_intra_background(self._interaction_matrix)

    def _compute_distance_matrix(self):
        self._current_distance_matrix = get_bayesian_edge_probs(
            self._interaction_matrix,
            self._inter_background, self._intra_background)

    def get_most_certain_edges(self, n):
        # score by diff between the best score for both nodesides, i.e. max at same row/col
        m = self._current_distance_matrix.data
        px(name="joining").imshow(m, title="distance matrix original")
        # find the element in matrix m that has biggest difference to the next lowest element in the same row and column
        next_lowest_on_rows = np.partition(m, 2, axis=1)[:, 2]
        next_lowest_on_columns = np.partition(m, 2, axis=0)[2, :]

        # tile to matrices
        next_lowest_on_rows = np.tile(next_lowest_on_rows, (len(m), 1)).T
        next_lowest_on_columns = np.tile(next_lowest_on_columns, (len(m), 1))
        lowest_in_general = np.minimum(next_lowest_on_rows, next_lowest_on_columns)

        # diffs
        diffs = lowest_in_general - m
        px(name="joining").imshow(diffs, title="diffs")

        best_indexes = np.unravel_index(np.argsort(diffs, axis=None)[::-1], diffs.shape)
        to_return = []
        nodes_picked = set()
        for indexes in zip(best_indexes[0], best_indexes[1]):
            nodeside_a = NodeSide.from_numeric_index(indexes[0])
            nodeside_b = NodeSide.from_numeric_index(indexes[1])
            node_a = nodeside_a.to_directed_node()
            node_b = nodeside_b.to_directed_node().reverse()  # reverse because we want l to be + direction for node b
            if node_a.node_id in nodes_picked or node_b.node_id in nodes_picked:
                continue

            if node_a.node_id == node_b.node_id:
                continue

            # do not pick if this edge is not the highest scoring among all edges from the two nodes
            nodeside_a_other_side = nodeside_a.other_side()
            nodeside_b_other_side = nodeside_b.other_side()
            all_scores = np.concatenate([m[nodeside_a.numeric_index, :], m[:, nodeside_a.numeric_index],
                                         m[nodeside_b.numeric_index, :], m[:, nodeside_b.numeric_index],
                                         m[nodeside_a_other_side.numeric_index, :], m[:, nodeside_a_other_side.numeric_index],
                                         m[nodeside_b_other_side.numeric_index, :], m[:, nodeside_b_other_side.numeric_index]
                                         ])
            if m[indexes[0], indexes[1]] > np.min(all_scores):
                logging.info(f"Skipping edge {node_a} -> {node_b} because it is not the best edge for both nodes. "
                             f"Best score: {np.min(all_scores)}, score: {m[indexes[0], indexes[1]]}")
                continue

            logging.info(f"   Best edge: {node_a} -> {node_b}, from indexes {indexes}")
            to_return.append((node_a, node_b))
            nodes_picked.add(node_a.node_id)
            nodes_picked.add(node_b.node_id)

            if len(to_return) >= n:
                break

        return to_return

    def run(self):
        n_original_contigs = self._interaction_matrix.n_contigs

        # merge the best edges
        for i in range(20):
            logging.info(f"Iteration {i}, {len(self._current_path)} nodes in path")
            n_contigs = self._interaction_matrix.n_contigs
            n_to_merge = n_contigs // 4 + 1

            # merge edges with good score and where the nodesides do not have good scores to other nodesides
            best_edges = self.get_most_certain_edges(n_to_merge)
            if len(best_edges) == 0:
                logging.info("No more edges to merge")
                break

            logging.info(f"Best edge: {best_edges}")
            logging.info(f"Current path before: {self._current_path}")

            new_path = []
            nodes_picked = set()
            for node_a, node_b in best_edges:
                # nodes here refer to indexes in the interaction matrix, with orientation
                # find the actual nodes in the current path and merge them
                id_a = node_a.node_id
                dir_a = node_a.orientation
                node_a = self._current_path[id_a]
                if dir_a == '-':
                    node_a = node_a.reverse()

                id_b = node_b.node_id
                dir_b = node_b.orientation
                node_b = self._current_path[id_b]
                if dir_b == '-':
                    node_b = node_b.reverse()

                new_path.append(CompundNode([node_a, node_b]))
                nodes_picked.add(id_a)
                nodes_picked.add(id_b)

            for i, node in enumerate(self._current_path):
                # skip those that have been merged
                if i not in nodes_picked:
                    new_path.append(node)

            print(f"New path: {new_path}")
            self._current_path = new_path

            if len(self._current_path) == 1:
                logging.info("Stopping because final path is one node")
                break

            # merge the best edges into single nodes

            # make a new path where modes in the best edges are together and the rest after that
            #logging.info(f"Path now is {self._current_path}")
            self._interaction_matrix = self._interaction_matrix.merge_edges(best_edges)
            #self._interaction_matrix.plot()
            #plt.show()
            self._compute_distance_matrix()

    def get_final_path(self):
        out = []
        for nodes in self._current_path:
            if isinstance(nodes, CompundNode):
                out.extend(nodes.flatten())
            else:
                out.append(nodes)

        return out
