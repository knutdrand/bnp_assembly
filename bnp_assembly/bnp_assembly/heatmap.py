import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.interaction_matrix import InteractionMatrix


def create_heatmap_figure(alignments: ScaffoldAlignments, bin_size: int, genome: bnp.Genome, locations_pair: GenomicLocationPair):
    bin_size = max(bin_size, genome.size // 1000, 1000)
    print("Using bin size", bin_size)
    interaction_matrix = InteractionMatrix.from_locations_pair(locations_pair, bin_size=bin_size)
    fig = interaction_matrix.plot()
    # add contig ids from agp file
    global_offset = genome.get_genome_context().global_offset
    scaffold_offsets = global_offset.get_offset(alignments.scaffold_id)
    contig_offsets = (scaffold_offsets + alignments.scaffold_start) // bin_size
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=contig_offsets, ticktext=alignments.contig_id.tolist())),
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )
    return fig, interaction_matrix
