from bionumpy.bnpdataclass import bnpdataclass


@bnpdataclass
class PairedReadPositions:
    contig_1: int
    position_1: int
    contig_2: int
    position_2: int
