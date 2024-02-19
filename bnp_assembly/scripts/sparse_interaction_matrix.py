import logging
logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
from bnp_assembly.input_data import FullInputData
from bnp_assembly.io import PairedReadStream
from bnp_assembly.make_scaffold import get_numeric_input_data
from bnp_assembly.sparse_interaction_matrix import NaiveSparseInteractionMatrix, BinnedNumericGlobalOffset, \
    SparseInteractionMatrix
import sys
import scipy
import plotly.express as px
import numpy as np


contig_file_name = sys.argv[1]
bam = sys.argv[2]
genome = bnp.Genome.from_file(contig_file_name)
read_stream = PairedReadStream.from_bam(genome, bam, mapq_threshold=20)
input_data = FullInputData(genome, read_stream)
contig_name_translation, numeric_input_data = get_numeric_input_data(input_data)

global_offset = BinnedNumericGlobalOffset.from_contig_sizes(numeric_input_data.contig_dict, 100)
print(global_offset._contig_sizes / global_offset._contig_n_bins)
matrix = SparseInteractionMatrix.from_reads(global_offset, numeric_input_data.location_pairs)
scipy.sparse.save_npz("testmatrix.npz", matrix.sparse_matrix)

#matrix = SparseInteractionMatrix(scipy.sparse.load_npz("testmatrix.npz"), global_offset)
#nonsparse = matrix.nonsparse_matrix
#fig = px.imshow(np.log2(nonsparse+1))
#fig.show()

