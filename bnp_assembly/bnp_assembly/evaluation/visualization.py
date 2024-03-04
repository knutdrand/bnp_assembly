from typing import List

from shared_memory_wrapper import from_file

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.io import PairedReadStream
from bnp_assembly.make_scaffold import get_numeric_contig_name_translation
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix
import bionumpy as bnp


def get_directed_nodes_from_scaffold_alignments(scaffold_alignments: ScaffoldAlignments, genome: bnp.Genome, limit_around_contig: None):
    contig_sizes, contig_name_translation = get_numeric_contig_name_translation(genome)
    name_to_id_translation = {v: k for k, v in contig_name_translation.items()}
    contigs = scaffold_alignments.contig_id
    contig_ids = [name_to_id_translation[contig.to_string()] for contig in contigs]
    directions = scaffold_alignments.orientation.tolist()
    nodes = [DirectedNode(contig_id, orientation) for contig_id, orientation in zip(contig_ids, directions)]

    if limit_around_contig is not None:
        contig_id = name_to_id_translation[limit_around_contig]
        contig_index = contig_ids.index(contig_id)
        nodes = nodes[max(0, contig_index - 1):min(contig_index + 2, len(nodes))]

    return nodes

def visualize_scaffolding_using_sparse_interaction_matrix(path: List[DirectedNode],
                                                          interaction_matrix: SparseInteractionMatrix):
    xaxis_names = [str(node) for node in path]
    path_matrix = interaction_matrix.get_matrix_for_path(path, as_raw_matrix=False)
    return path_matrix.plot(xaxis_names=xaxis_names)


def visualize_from_agp(contig_filename, agp_filename, sparse_interaction_matrix_filename, around_contig=None):
    matrix = from_file(sparse_interaction_matrix_filename)
    agp = ScaffoldAlignments.from_agp(agp_filename)
    genome = bnp.Genome.from_file(contig_filename)
    nodes = get_directed_nodes_from_scaffold_alignments(agp, genome, around_contig)
    return visualize_scaffolding_using_sparse_interaction_matrix(nodes, matrix)

