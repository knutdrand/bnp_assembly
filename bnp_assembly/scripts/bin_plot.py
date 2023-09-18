import bionumpy as bnp
import numpy as np
from matplotlib import pyplot as plt

read_path = '../../benchmarking/data/athalia_rosea/real/big/10/10000000/1/not_assembled/0/200/hifiasm.hic.p_ctg.bam'
fa_path = '../../benchmarking/data/athalia_rosea/real/big/10/10000000/1/not_assembled/0/200/hifiasm.hic.p_ctg.fa'
bg = bnp.genomic_data.BinnedGenome.from_file(fa_path, bin_size=1000)
chunks = bnp.open(read_path).read_chunks()
for i, chunk in enumerate(chunks):
    print(i)
    bg.count(chunk[chunk.mapq > 20])
d = bg.count_dict
print(d)

# def find_read_counts(binned, n_reads, n_read_bins=10):
#     return np.searchsorted(np.cumsum(binned), n_reads*np.arange(n_read_bins))
#
# n_reads=100
# n_read_bins=20
# x = n_reads * np.arange(n_read_bins)
# for v in d.values():
#     plt.plot(x, find_read_counts(v, n_reads, n_read_bins))
# plt.show()
