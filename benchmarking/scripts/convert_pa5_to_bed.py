import logging
logging.basicConfig(level=logging.INFO)
import sys

pa5 = sys.argv[1]

with open(pa5) as f:
    for i, line in enumerate(f):
        if line.startswith('#'):
            continue

        if i % 100000 == 0:
            logging.info(f"Converted {i} pa5 lines")

        line = line.strip().split()
        name = line[0]
        contig1 = line[1]
        pos1 = int(line[2])
        contig2 = line[3]
        pos2 = int(line[4])
        strand1 = line[5]
        strand2 = line[6]
        line1 = [contig1, pos1, pos1+100, name + "/1", 60, strand1]
        line2 = [contig2, pos2, pos2+100, name + "/2", 60, strand2]

        print("\t".join(map(str, line1)))
        print("\t".join(map(str, line2)))
