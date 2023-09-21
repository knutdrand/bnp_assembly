import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.interaction_matrix import InteractionMatrix
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison


def create_heatmap_figure(alignments: ScaffoldAlignments, bin_size: int, genome: bnp.Genome,
                          locations_pair: GenomicLocationPair, true_alignments: ScaffoldAlignments = None):
    bin_size = max(bin_size, genome.size // 1000, 1000)
    print("Using bin size", bin_size)
    interaction_matrix = InteractionMatrix.from_locations_pair(locations_pair, bin_size=bin_size)
    fig = interaction_matrix.plot()
    # add contig ids from agp file
    global_offset = genome.get_genome_context().global_offset
    scaffold_offsets = global_offset.get_offset(alignments.scaffold_id)
    contig_offsets = (scaffold_offsets + alignments.scaffold_start) // bin_size
    contig_names = alignments.contig_id.tolist()
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=contig_offsets, ticktext=contig_names)),
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )

    if true_alignments is not None:
        contig_offsets = {c: n for c, n in zip(contig_names, contig_offsets)}
        print("Contig offsets", contig_offsets)
        comparer = ScaffoldComparison(alignments, true_alignments)
        wrong_edges = comparer.false_edges()
        for edge in wrong_edges:
            # to_node_side is last node, show marker at beginning this
            heatmap_pos = contig_offsets[edge.to_node_side.node_id]
            print("  Wrong edge %s at %d" % (edge, heatmap_pos))
            fig.add_vline(x=heatmap_pos, line_width=1, line_dash="dash", line_color="white")

    return fig, interaction_matrix
