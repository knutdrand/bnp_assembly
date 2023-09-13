from bionumpy.bnpdataclass import bnpdataclass


@bnpdataclass
class PairedReadPositions:
    contig_1: str
    position_1: int
    contig_2: str
    position_2: int
