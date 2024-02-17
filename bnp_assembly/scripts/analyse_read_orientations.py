import sys
import bionumpy as bnp
import numpy as np

file_name = sys.argv[1]
file = bnp.open(file_name)

chunk = file.read_chunk(min_chunk_size=1000000000)

reads1 = chunk[0::2]
reads2 = chunk[1::2]

print(reads1.name)
print(reads2.name)

assert np.all(reads1.name == reads2.name)

flags1 = reads1.flag - 64 - 1
flags2 = reads2.flag - 128 - 1


types = {
    "both_reverse": 48,
    "first_reverse": 16,
    "second_reverse": 32,
    "both_forward": 0
}


n_tot = len(flags1)
print(np.unique(flags1))
for type, flag in types.items():
    n_type = np.sum(flags1 == flag)
    print(type, n_type, n_type / n_tot)