import bionumpy as bnp
import numpy as np

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.generate_training import generate_training_data_set
from bnp_assembly.io import PairedReadStream
from bnp_assembly.scaffolds import Scaffolds


def test_generate_training_data_set():
    dir = "example_data/yeast_small/"
    truth = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(dir + "hifiasm.hic.p_ctg.agp"))
    genome = bnp.Genome.from_file(dir + "hifiasm.hic.p_ctg.fa")
    reads = PairedReadStream.from_bam(genome,
                                      dir + "hifiasm.hic.p_ctg.sorted_by_read_name.bam",
                                      mapq_threshold=10)
    data_set = generate_training_data_set(truth, genome, reads)
    assert np.any(data_set['y']), data_set
