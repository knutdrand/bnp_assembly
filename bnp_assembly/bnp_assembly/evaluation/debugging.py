import time
from typing import Union, Iterable

import numpy as np
from .. import plotting
import bionumpy as bnp

from ..contig_graph import DirectedNode
from ..graph_objects import NodeSide
from ..io import PairedReadStream
from ..location import LocationPair
from ..make_scaffold import get_numeric_contig_name_translation
from ..scaffolds import Scaffolds
import logging


class ScaffoldingDebugger:
    def __init__(self,
                 estimated_scaffolds: Scaffolds,
                 truth_scaffolds: Scaffolds,
                 contigs: bnp.Genome,
                 reads: PairedReadStream,
                 plotting_folder: str = "./"):
        self.estimated_scaffolds = estimated_scaffolds
        self.truth_scaffolds = truth_scaffolds
        print("True path")
        print(self.truth_scaffolds._scaffolds)
        print("True scaffolds")
        print(self.truth_scaffolds.edges)
        print("Estimated scaffolds")
        print(self.estimated_scaffolds.edges)
        self.contigs = contigs
        self._read_stream = reads

        plotting.register(debug=plotting.ResultFolder(plotting_folder))
        self.px = plotting.px(name="debug")

        contig_sizes, numeric_to_name_translation = get_numeric_contig_name_translation(self.contigs)
        self.contig_name_translation = {val: key for key, val in numeric_to_name_translation.items()}
        logging.info("Contig sizes: %s" % contig_sizes)
        self.contig_sizes = contig_sizes

    def get_reads_for_contig(self, contig_name):
        numeric_contig_name = self.contig_name_translation[contig_name]
        reads = next(self._read_stream)

        return LocationPair.from_multiple_location_pairs([
            chunk.filter_on_contig(numeric_contig_name) for chunk in reads
        ])

    def get_reads_between_contigs(self, contig_a, contig_b):
        contig_a = self.contig_name_translation[contig_a]
        contig_b = self.contig_name_translation[contig_b]
        reads = next(self._read_stream)

        return LocationPair.from_multiple_location_pairs([
            chunk.filter_on_two_contigs(contig_a, contig_b)
            for chunk in reads
        ])

    def make_heatmap_for_two_contigs(self, node_a: DirectedNode, node_b: DirectedNode, bin_size=10000):
        contig_a = node_a.node_id
        contig_b = node_b.node_id

        contig_a_id = self.contig_name_translation[contig_a]
        contig_b_id = self.contig_name_translation[contig_b]
        reads_between = self.get_reads_between_contigs(contig_a, contig_b)

        heatmap_size = self.contig_sizes[contig_a_id] + self.contig_sizes[contig_b_id]
        total_contig_sizes = self.contig_sizes[contig_a_id] + self.contig_sizes[contig_b_id]
        n_bins = heatmap_size // bin_size

        if n_bins > 5000:
            bin_size = total_contig_sizes // 5000
            logging.info("Adjusting bin size to %d", bin_size)
        elif n_bins < 100:
            bin_size = total_contig_sizes // 100

        heatmap = np.zeros((heatmap_size // bin_size + 1, heatmap_size // bin_size + 1))
        logging.info("IN total %d reads between nodes" % (len(reads_between.location_a)))
        for read_a, read_b in zip(reads_between.location_a, reads_between.location_b):
            pos_a = read_a.offset
            pos_b = read_b.offset

            if node_a.orientation == "-":
                pos_a = self.contig_sizes[int(read_a.contig_id)] - pos_a

            if node_b.orientation == "-":
                pos_b = self.contig_sizes[int(read_b.contig_id)] - pos_b

            if read_a.contig_id == contig_b_id:
                pos_a += self.contig_sizes[contig_a_id]

            if read_b.contig_id == contig_b_id:
                pos_b += self.contig_sizes[contig_a_id]

            heatmap[pos_a // bin_size, pos_b // bin_size] += 1
            heatmap[pos_b // bin_size, pos_a // bin_size] += 1

        fig = self.px.imshow(np.log2(heatmap + 1), title=f"Heatmap for {node_a} and {node_b}")

        # add contigs
        contig_offsets = [0, self.contig_sizes[contig_a_id] // bin_size]
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=contig_offsets, ticktext=[contig_a, contig_b]),
            yaxis=dict(tickmode='array', tickvals=contig_offsets, ticktext=[contig_a, contig_b]),
        )
        fig.update_xaxes(
            showgrid=True,
            ticks="outside",
            tickson="boundaries",
            ticklen=20
        )
        fig.show()
        return fig

    def debug_edge(self, contig_a: DirectedNode, contig_b: DirectedNode):
        self.make_heatmap_for_two_contigs(contig_a, contig_b)

        contig_a_neighbour = self.truth_scaffolds.get_neighbour(contig_a)
        if contig_a_neighbour:
            logging.info(f"   Contig {contig_a} should be linked to {contig_a_neighbour}")
            self.make_heatmap_for_two_contigs(contig_a, contig_a_neighbour)

        contig_b_neighbour = self.truth_scaffolds.get_neighbour(contig_b)
        if contig_b_neighbour:
            logging.info(f"   Contig {contig_b} should be linked to {contig_b_neighbour}")
            self.make_heatmap_for_two_contigs(contig_b, contig_b_neighbour)

    def debug_wrong_edges(self):
        i = 0
        for edge in self.estimated_scaffolds.edges:
            if edge not in self.truth_scaffolds.edges:
                contig_a = DirectedNode(edge.from_node_side.node_id, "+" if edge.from_node_side.side == "r" else "-")
                contig_b = DirectedNode(edge.to_node_side.node_id, "+" if edge.to_node_side.side == "l" else "-")
                logging.info("False edge between %s and %s" % (contig_a, contig_b))
                self.debug_edge(contig_a, contig_b)
                i += 1
                if i >= 3:
                    break


    def finish(self):
        self.px.write_report()
