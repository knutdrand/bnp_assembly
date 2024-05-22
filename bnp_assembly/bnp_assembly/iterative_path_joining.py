import logging
import random
import time
from typing import Union, List, Dict

import matspy
import numpy as np
import plotly
import scipy
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_path_optimization import split_using_inter_distribution, get_splitting_edge_scores
from matplotlib import pyplot as plt

from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.graph_objects import NodeSide, Edge
from bnp_assembly.sparse_interaction_based_distance import get_bayesian_edge_probs, get_inter_background, \
    get_intra_background, get_intra_distance_background, get_background_fixed_distance, \
    get_inter_as_mix_between_inside_outside, get_inter_as_mix_between_inside_outside_multires, get_intra_as_mix
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix, BackgroundInterMatrices, \
    BinnedNumericGlobalOffset
from bnp_assembly.plotting import px

class CompundNode:
    def __init__(self, nodes: List[Union[DirectedNode, 'CompundNode']]):
        self.nodes = nodes

    def reverse(self):
        return CompundNode([node.reverse() for node in self.nodes][::-1])

    def __str__(self):
        return f"C[{self.nodes}]"

    def __repr__(self):
        return str(self)

    def append(self, node):
        self.nodes.append(node)

    def flatten(self) -> List[Union[DirectedNode, 'CompundNode']]:
        to_return = []
        for node in self.nodes:
            if isinstance(node, CompundNode):
                to_return.extend(node.flatten())
            else:
                to_return.append(node)
        return to_return

    def contains_edge(self, edge: Edge):
        for prev_node, next_node in zip(self.nodes[:-1], self.nodes[1:]):
            if edge.from_node_side == prev_node.right_side and edge.to_node_side == next_node.left_side:
                return True
        return False

    def split_on_edges(self, edges: List[Edge]):
        """Splits on edges (if they exist in node) and returns a list of CompundNodes.
        Assumes this compundnode only contains DirectedNode, i.e. not nested"""
        assert all([isinstance(node, DirectedNode) for node in self.nodes])
        path_edges = [Edge(n1.right_side, n2.left_side) for n1, n2 in zip(self.nodes[:-1], self.nodes[1:])]
        split_indexes = [i for i, edge in enumerate(path_edges) if edge in edges]
        if len(split_indexes) == 0:
            return [self]
        return self.split_on_edge_indexes(split_indexes)

    def split_on_edge_indexes(self, split_indexes):
        out = []
        current = []
        for i, node in enumerate(self.nodes):
            current.append(node)
            if i in split_indexes:
                out.append(CompundNode(current))
                current = []
        if len(current) > 0:
            out.append(CompundNode(current))
        return out


class IterativePathJoiner:
    def __init__(self, interaction_matrix: SparseInteractionMatrix, skip_init_distance_matrix=False):
        self._interaction_matrix = interaction_matrix
        self._original_interaction_matrix = interaction_matrix
        self._original_global_offset = interaction_matrix.global_offset
        self._current_path = []
        self._current_distance_matrix = None
        self._inter_background = None
        self._inter_background_means = None
        self._inter_background_stds = None
        self._intra_background = None
        self._skip_init_distance_matrix = skip_init_distance_matrix
        self.shuffle() # start with random path

    def shuffle(self):
        n_contigs = self._interaction_matrix.n_contigs
        initial_path = [DirectedNode(contig, '+') for contig in range(n_contigs)]
        random.seed(0)
        random.shuffle(initial_path)
        self._current_path = initial_path
        self._interaction_matrix = self._interaction_matrix.get_matrix_for_path2(initial_path, as_raw_matrix=False, backend=scipy.sparse.csr_matrix)
        self._get_backgrounds()
        if not self._skip_init_distance_matrix:
            self._compute_distance_matrix()

    @property
    def current_interaction_matrix(self):
        return self._interaction_matrix

    def _get_backgrounds(self):
        # n_samples and max gins for inter
        matrix = self._interaction_matrix
        if isinstance(self._interaction_matrix.sparse_matrix, scipy.sparse.coo_matrix):
            matrix.to_csr_matrix()

        self._inter_background_means, self._inter_background_stds = (
            get_inter_as_mix_between_inside_outside_multires(matrix, 25, 5000, ratio=0.5))

        logging.info("Getting intra..")
        intra0 = get_intra_as_mix(matrix, 20, 5000)

        lowest_size = min(intra0.shape[1], self._inter_background_means.shape[0])
        logging.info(f"Lowest size: {lowest_size}")
        intra0 = intra0[:, :lowest_size, :lowest_size]
        self._inter_background_means = self._inter_background_means[:lowest_size, :lowest_size]
        self._inter_background_stds = self._inter_background_stds[:lowest_size, :lowest_size]

        self._intra_background = intra0  # np.concatenate([intra0], axis=0)
        self._intra_background_means = self._intra_background.mean(axis=0)
        self._intra_background_stds = self._intra_background.std(axis=0)

        inter2 = get_inter_background(matrix, 20, 5000)
        self._inter_background_means2 = inter2.mean(axis=0)
        self._inter_background_stds2 = inter2.std(axis=0)
        #self._inter_background_means2, self._inter_background_stds2 = (
        #    get_inter_as_mix_between_inside_outside_multires(matrix, 20, 5000, ratio=1.0, distance_type="close"))

        self._intra_background_means2, self._intra_background_stds2 = (
            get_inter_as_mix_between_inside_outside_multires(matrix, 20, 5000, ratio=0.95, distance_type="close"))

        #intra = get_background_fixed_distance(matrix, 200, 5000, "outer_contig")
        #self._intra_background_means2 = intra.mean(axis=0)
        #self._intra_background_stds2 = intra.std(axis=0)


        self._intra_background_means2 = self._intra_background_means2[:lowest_size, :lowest_size]
        self._intra_background_stds2 = self._intra_background_stds2[:lowest_size, :lowest_size]
        self._inter_background_means2 = self._inter_background_means2[:lowest_size, :lowest_size]
        self._inter_background_stds2 = self._inter_background_stds2[:lowest_size, :lowest_size]


    def split(self, n_to_split):
        n_to_split = n_to_split
        if n_to_split == 0:
            return
        # split the worst edges that we have joined so far
        path = self.get_final_path()
        contig_path = ContigPath.from_directed_nodes(path)
        path_matrix = self._original_interaction_matrix.get_matrix_for_path2(path, as_raw_matrix=False)
        inter_background = BackgroundInterMatrices.from_sparse_interaction_matrix(path_matrix)
        edge_scores = get_splitting_edge_scores(path_matrix, inter_background, contig_path)
        edge_scores = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        n_splitted = 0
        logging.info(f"Edge scores: {edge_scores[0:10]}")

        # find which edges are inside mered nodes, pick these to split on
        edges_to_split = []
        for edge, score in edge_scores:
            if len(edges_to_split) == n_to_split:
                break
            for node in self._current_path:
                if isinstance(node, CompundNode) and node.contains_edge(edge):
                    edges_to_split.append(edge)
                    n_splitted += 1
                    break

        logging.info(f"Splitting on {edges_to_split[0:10]}")

        # do the actual splitting
        new_path = []
        for node in self._current_path:
            if isinstance(node, DirectedNode):
                new_path.append(node)
            else:
                new_path.extend(node.split_on_edges(edges_to_split))

        logging.info(f"Now path is {new_path[0:50]}")
        new_contig_sizes = []
        new_contig_n_bins = []
        for nodes in new_path:
            if isinstance(nodes, DirectedNode):
                new_contig_sizes.append(self._original_global_offset.contig_sizes[nodes.node_id])
                new_contig_n_bins.append(self._original_global_offset.contig_n_bins[nodes.node_id])
            else:
                assert isinstance(nodes, CompundNode)
                new_contig_sizes.append(sum([self._original_global_offset.contig_sizes[node.node_id] for node in nodes.nodes]))
                new_contig_n_bins.append(sum([self._original_global_offset.contig_n_bins[node.node_id] for node in nodes.nodes]))

        new_bin_offsets = np.cumsum(np.insert(new_contig_n_bins, 0, 0))
        new_global_offset = BinnedNumericGlobalOffset(np.array(new_contig_sizes), np.array(new_contig_n_bins),
                                                      new_bin_offsets)

        self._current_path = new_path
        self._interaction_matrix.set_global_offset(new_global_offset)
        self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        t0 = time.perf_counter()
        self._current_distance_matrix = get_bayesian_edge_probs(
            self._interaction_matrix,
            self._inter_background_means,
            self._inter_background_stds,
            self._intra_background_means,
            self._intra_background_stds,
        )
        logging.info("Time to compute distance matrix: %.2f" % (time.perf_counter() - t0))

        self._current_distance_matrix2 = get_bayesian_edge_probs(
            self._interaction_matrix,
            self._inter_background_means2,
            self._inter_background_stds2,
            self._intra_background_means2,
            self._intra_background_stds2,
        )

    def init_with_scaffold_alignments(self, scaffold_alignments: ScaffoldAlignments, contig_names_to_ids: Dict[str, int]):
        """Initiates by reading scaffolds from agp and starting with all contigs inside scaffolds joined"""
        scaffold_edges = scaffold_alignments.to_list_of_edges()
        # translate to numeric ids

        path = scaffold_alignments.get_list_of_nodes()
        # translate to numeric ids
        path = [
            [DirectedNode(contig_names_to_ids[node.node_id], node.orientation) for node in nodes]
            for nodes in path
        ]
        self._interaction_matrix = self._interaction_matrix.get_matrix_for_path2(
            [node for nodes in path for node in nodes], as_raw_matrix=False
        )
        self._interaction_matrix.set_global_offset(self._original_global_offset.get_new_by_merging_nodes(
            [[n.node_id for n in nodes] for nodes in path]
        ))

        path = [CompundNode(nodes) if len(nodes) > 1 else nodes[0] for nodes in path]
        logging.info("Initing with path")
        logging.info(path)
        self._current_path = path
        logging.info(f"{len(path)} nodes after initing")
        self._get_backgrounds()
        self._compute_distance_matrix()

    def get_most_certain_edges(self, n):
        t0 = time.perf_counter()
        # score by diff between the best score for both nodesides, i.e. max at same row/col
        m = self._current_distance_matrix.data
        #plotly.express.imshow(m).show()
        #self._interaction_matrix.plot()
        #plt.show()
        # find the element in matrix m that has biggest difference to the next lowest element in the same row and column
        next_lowest_on_rows = np.partition(m, 2, axis=1)[:, 2]
        next_lowest_on_columns = np.partition(m, 2, axis=0)[2, :]

        lowest_on_rows = np.min(m, axis=1)
        logging.info(f"This many has same lowest and next lowest: {np.sum(next_lowest_on_rows == lowest_on_rows)}")

        # tile to matrices
        next_lowest_on_rows = np.tile(next_lowest_on_rows, (len(m), 1)).T
        next_lowest_on_columns = np.tile(next_lowest_on_columns, (len(m), 1))
        lowest_in_general = np.minimum(next_lowest_on_rows, next_lowest_on_columns)

        # diffs
        diffs = lowest_in_general - m
        #px(name="joining").imshow(diffs, title="diffs")

        # look among the best diffs
        #best_indexes = np.unravel_index(np.argsort(diffs, axis=None)[::-1][:n*100], diffs.shape)
        best_indexes = np.unravel_index(np.argsort(diffs, axis=None)[::-1], diffs.shape)
        to_return = []
        nodes_picked = set()
        for indexes in zip(best_indexes[0], best_indexes[1]):
            nodeside_a = NodeSide.from_numeric_index(indexes[0])
            nodeside_b = NodeSide.from_numeric_index(indexes[1])
            score = m[indexes[0], indexes[1]]
            score2 = self._current_distance_matrix2.data[indexes[0], indexes[1]]

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
                #logging.info(f"Skipping edge {node_a} -> {node_b} because it is not the best edge for both nodes. "
                #             f"Best score: {np.min(all_scores)}, score: {m[indexes[0], indexes[1]]}")
                continue

            if score2 > -np.log(0.9) and True:
                logging.info(f"Ignoring {node_a} -> {node_b} because score is {score2} ({score})")
                continue

                #logging.info(f"   Best edge: {node_a} -> {node_b}, from indexes {indexes}")
            to_return.append((node_a, node_b))
            nodes_picked.add(node_a.node_id)
            nodes_picked.add(node_b.node_id)

            if len(to_return) >= n:
                break

        logging.info("Time to find best edges: %.2f" % (time.perf_counter() - t0))
        return to_return

    def run(self, n_rounds=100):
        n_original_contigs = self._interaction_matrix.n_contigs

        # merge the best edges
        n_joined_prev_iteration = 0
        for i in range(n_rounds):
            if i == 0:
                m = self._current_distance_matrix
                #plotly.express.imshow(m.data[::2, 1::2], title="distance matrix").show()

            n_contigs = self._interaction_matrix.n_contigs
            n_to_merge = n_contigs // 3 + 1
            n_to_split = n_joined_prev_iteration // 3 - 1
            logging.info("")
            logging.info(f"Iteration {i}, {len(self._current_path)} nodes in path. Will split {n_to_split} and merge {n_to_merge} nodes.")
            if i > 0 and n_to_split > 0 and False:  #i % 2 == 1:
                self.split(n_to_split)

            logging.info(f"There are {len(self._current_path)} nodes after splitting.")

            # merge edges with good score and where the nodesides do not have good scores to other nodesides
            best_edges = self.get_most_certain_edges(n_to_merge)
            if len(best_edges) == 0:
                logging.info("No more edges to merge")
                break

            logging.info(f"Best edge: {best_edges[0:50]}")

            n_joined_prev_iteration = len(best_edges)

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

            for j, node in enumerate(self._current_path):
                # skip those that have been merged
                if j not in nodes_picked:
                    new_path.append(node)

            #logging.info(f"New path: {new_path[0:30]}")
            self._current_path = new_path
            self._cleanup_path()

            if len(self._current_path) == 1:
                logging.info("Stopping because final path is one node")
                break

            # merge the best edges into single nodes

            # make a new path where modes in the best edges are together and the rest after that
            #logging.info(f"Path now is {self._current_path[0:50]}..")
            logging.info(f"Length of path is {len(self._current_path)}")
            self._interaction_matrix = self._interaction_matrix.merge_edges(best_edges)
            #self._interaction_matrix.plot()
            if i % 2 == 0:
                logging.info("Getting backroungds")
                self._get_backgrounds()
            else:
                logging.info(f"Not getting backgrounds, {i}")
            self._compute_distance_matrix()
            #plt.show()

    def _cleanup_path(self):
        """Flatten nested nodes"""
        new_path = []
        for node in self._current_path:
            if isinstance(node, CompundNode):
                new_path.append(CompundNode(node.flatten()))
            else:
                new_path.append(node)
        self._current_path = new_path

    def get_final_path(self) -> List[DirectedNode]:
        out = []
        for nodes in self._current_path:
            if isinstance(nodes, CompundNode):
                out.extend(nodes.flatten())
            else:
                out.append(nodes)

        return out

    def get_final_path_as_list_of_contigpaths(self) -> List[ContigPath]:
        out = []
        for nodes in self._current_path:
            if isinstance(nodes, CompundNode):
                out.append(ContigPath.from_directed_nodes(nodes.flatten()))
            else:
                out.append(ContigPath.from_directed_nodes([nodes]))
        return out
