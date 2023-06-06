from collections import defaultdict
from typing import Dict, List

from bionumpy.bnpdataclass import bnpdataclass


@bnpdataclass
class ScaffoldAlignments:
    scaffold_id: str
    scaffold_start: int
    scaffold_end: int
    contig_id: str
    contig_start: int
    contig_end: int
    orientation: str

    def to_agp(self, file_name):
        with open(file_name, "w") as f:
            counters = defaultdict(lambda: 1)
            for entry in self:
                f.write(f"{entry.scaffold_id.to_string()}\t{entry.scaffold_start+1}\t{entry.scaffold_end+1}\t"
                        f"{counters[entry.scaffold_id.to_string()]}\tW\t{entry.contig_id.to_string()}"
                        f"\t{entry.contig_start+1}\t{entry.contig_end+1}\t{entry.orientation.to_string()}\n")
                counters[entry.scaffold_id.to_string()] += 1

    @classmethod
    def from_agp(cls, file_name):
        entries = []
        with open(file_name) as f:
            for line in f:
                line = line.strip().split()
                entries.append(
                    (line[0],
                    int(line[1])-1,
                    int(line[2])-1,
                    line[5],
                    int(line[6])-1,
                    int(line[7])-1,
                    line[8])
                )
        return cls.from_entry_tuples(entries)
