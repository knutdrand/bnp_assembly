from typing import List
from bnp_assembly.io import PairedReadStream


def show_contigs_heatmap(contig_names: List[str], read_stream: PairedReadStream, out_path: str):
    """Show heatmap for only specific contigs together"""