import logging

import bionumpy as bnp
import pandas as pd

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.cli import app
from bnp_assembly.graph_objects import Edge, NodeSide
from bnp_assembly.input_data import NumericInputData
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.io import get_genomic_read_pairs, PairedReadStream
from bnp_assembly.make_scaffold import get_numeric_contig_name_translation, create_distance_matrix_from_reads
from bnp_assembly.scaffolds import Scaffolds


def generate_training_data_set(truth, genome, reads) -> pd.DataFrame:
    contig_sizes, contig_name_translation = get_numeric_contig_name_translation(genome)
    feature_matrix = create_distance_matrix_from_reads(NumericInputData(contig_sizes, reads))
    rows = []
    edge_names = {str(edge) for edge in truth.edges}
    print(edge_names)
    for edge in feature_matrix.keys():
        translated_edge = Edge(NodeSide(contig_name_translation[edge.from_node_side.node_id], edge.from_node_side.side),
                               NodeSide(contig_name_translation[edge.to_node_side.node_id], edge.to_node_side.side))
        y = (translated_edge in truth.edges) or (translated_edge.reverse() in truth.edges)
        x = feature_matrix[edge]
        rows.append([str(translated_edge), x, y])
    return pd.DataFrame(rows, columns=['edge', "x", "y"])


@app.command()
def generate_training(contig_file_name: str, read_filename: str, true_agp_path, out_file_name: str):
    truth = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(true_agp_path))
    genome = bnp.Genome.from_file(contig_file_name, filter_function=None)
    reads = PairedReadStream.from_bam(genome, read_filename, mapq_threshold=20)
    data_set = generate_training_data_set(truth, genome, reads)
    data_set.write(out_file_name)
