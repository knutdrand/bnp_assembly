from typing import Iterable, List
import logging
import numpy as np
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering

from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.location import LocationPair

logger = logging.getLogger(__name__)

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.show()


def create_similarity_matrix(contig_sizes: Iterable, read_pairs: Iterable[LocationPair]):
    counter = count_interactions(contig_sizes, read_pairs)
    counter += 1
    np.fill_diagonal(counter, 0)
    counter += counter.T
    counter = counter/(counter.sum(axis=0, keepdims=True)*counter.sum(axis=1, keepdims=True))
    return counter/np.max(counter)


def count_interactions(contig_sizes, read_pairs):
    n_contigs = len(contig_sizes)
    counter = sum(count_contig_pairs(read_pair, n_contigs) for read_pair in read_pairs)
    return counter


def count_contig_pairs(read_pair, n_contigs):
    assert isinstance(read_pair, LocationPair), read_pair
    shape = (n_contigs, n_contigs)
    flat_indices = np.ravel_multi_index((read_pair.location_a.contig_id, read_pair.location_b.contig_id), shape)
    return np.bincount(flat_indices, minlength=n_contigs ** 2).reshape(shape)

def hierarchical_cluster(similarity_matrix):
    distance_matrix = 1 - similarity_matrix
    model = AgglomerativeClustering(affinity="precomputed", linkage='average', compute_distances=True)
    model.fit(distance_matrix)
    plot_dendrogram(model, truncate_mode='level', p=3)
    return model.labels_
    #
    # from scipy.cluster.hierarchy import linkage, leaves_list
    # linkage_matrix = linkage(distance_matrix, method='ward')
    # print(linkage_matrix)
    # return leaves_list(linkage_matrix)


def split_based_on_labels(labels: np.ndarray, path: ContigPath)-> List[ContigPath]:
    logger.info(f'path: {path}')
    logger.info(f'labels: {labels}')
    split_edges = [edge for edge in path.edges if labels[edge.from_node_side.node_id]!=labels[edge.to_node_side.node_id]]
    logger.info(f'split_edges: {split_edges}')
    return path.split_on_edges(split_edges)


def cluster_split(numeric_input_data, path: ContigPath):
    distance_matrix = create_similarity_matrix(numeric_input_data.contig_dict, next(numeric_input_data.location_pairs))
    px.imshow(distance_matrix).show()
    labels = hierarchical_cluster(distance_matrix)
    logger.info(f'labels: {labels}')
    return split_based_on_labels(labels, path)



