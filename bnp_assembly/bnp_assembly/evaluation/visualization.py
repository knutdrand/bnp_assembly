import logging
from typing import List

from shared_memory_wrapper import from_file

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.io import PairedReadStream
from bnp_assembly.make_scaffold import get_numeric_contig_name_translation
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix
import bionumpy as bnp


def get_directed_nodes_from_scaffold_alignments(scaffold_alignments: ScaffoldAlignments, genome: bnp.Genome, limit_around_contig: None, window_size=1):
    contig_sizes, contig_name_translation = get_numeric_contig_name_translation(genome)
    name_to_id_translation = {v: k for k, v in contig_name_translation.items()}
    print(name_to_id_translation)
    contigs = scaffold_alignments.contig_id
    contig_ids = [name_to_id_translation[contig.to_string()] for contig in contigs]
    directions = scaffold_alignments.orientation.tolist()
    nodes = [DirectedNode(contig_id, orientation) for contig_id, orientation in zip(contig_ids, directions)]
    nonnumeric_contig_names = [contig.to_string() for contig in contigs]

    if limit_around_contig is not None:
        contig_id = name_to_id_translation[limit_around_contig]
        logging.info("Limiting to contig %s, id %d" % (limit_around_contig, contig_id))
        contig_index = contig_ids.index(contig_id)
        nodes = nodes[max(0, contig_index - window_size):min(contig_index + window_size + 1, len(nodes))]
        logging.info(f"Numeric ids are: {nodes}")
        nonnumeric_contig_names = nonnumeric_contig_names[max(0, contig_index - window_size):min(contig_index + window_size + 1, len(contigs))]

    return nodes, nonnumeric_contig_names


def visualize_scaffolding_using_sparse_interaction_matrix(path: List[DirectedNode],
                                                          interaction_matrix: SparseInteractionMatrix,
                                                          xaxis_names=None):
    #logging.info("Visualizing for path: %s" % xaxis_names)
    logging.info("Getting path matrix")
    if len(path) == interaction_matrix.n_contigs:
        # v2 only works when path covers all contigs, but is much faster
        path_matrix = interaction_matrix.get_matrix_for_path2(path, as_raw_matrix=False)
    else:
        logging.info("Subpath")
        path_matrix = interaction_matrix.get_matrix_for_path(path, as_raw_matrix=False)
    #logging.info(f"xaxis_names: {xaxis_names}")

    return path_matrix.plot(xaxis_names=xaxis_names), path_matrix


def visualize_from_agp(contig_filename, agp_filename, sparse_interaction_matrix_filename, around_contig=None):
    matrix = from_file(sparse_interaction_matrix_filename)
    agp = ScaffoldAlignments.from_agp(agp_filename)
    genome = bnp.Genome.from_file(contig_filename)
    nodes, nonumeric_nodes = get_directed_nodes_from_scaffold_alignments(
        agp, genome, around_contig, window_size=2)
    return visualize_scaffolding_using_sparse_interaction_matrix(nodes, matrix, nonumeric_nodes)

