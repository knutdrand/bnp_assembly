import logging
logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
from bnp_assembly.input_data import FullInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.make_scaffold import get_numeric_input_data
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix
import sys
import scipy


contig_file_name = sys.argv[1]
bam = sys.argv[2]
genome = bnp.Genome.from_file(contig_file_name)
read_stream = PairedReadStream.from_bam(genome, bam, mapq_threshold=20)
input_data = FullInputData(genome, read_stream)
contig_name_translation, numeric_input_data = get_numeric_input_data(input_data)
matrix = SparseInteractionMatrix.from_reads(numeric_input_data.contig_dict, numeric_input_data.location_pairs, 100)

scipy.sparse.save_npz("testmatrix.npz", matrix.sparse_matrix)