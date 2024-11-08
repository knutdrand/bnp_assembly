import logging
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
from bionumpy import LocationEntry
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.encoded_array import ASCIIEncoding
from bionumpy.genomic_data.global_offset import GlobalOffset
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.graph_objects import Edge

from bnp_assembly.location import Location
from bnp_assembly.simulation.pair_distribution import PairedLocationEntry
import bionumpy as bnp


class ScaffoldMap:
    def __init__(self, scaffold_alignments):
        self._scaffold_alignments = scaffold_alignments
        self._starts = defaultdict(list)
        self._ends = defaultdict(list)
        self._contig_ids = defaultdict(list)
        self._contig_dict = {}
        self._scaffold_offsets = {}
        self._all_positive = True
        for i, alignment in enumerate(scaffold_alignments):
            self._starts[str(alignment.scaffold_id)].append(int(alignment.scaffold_start))
            self._ends[str(alignment.scaffold_id)].append(int(alignment.scaffold_end))
            self._contig_ids[str(alignment.scaffold_id)].append(str(alignment.contig_id))
            self._contig_dict[str(alignment.contig_id)] = int(alignment.contig_end)-int(alignment.contig_start)
            self._scaffold_offsets[str(alignment.contig_id)] = (str(alignment.scaffold_id), int(alignment.scaffold_start), str(alignment.orientation))
            if not alignment.orientation == '+':
                self._all_positive = False
            assert alignment.contig_start == 0

        starts = {scaffold_id: np.array(self._starts[scaffold_id]) for scaffold_id in self._starts}
        ends = {scaffold_id: np.array(self._ends[scaffold_id]) for scaffold_id in self._ends}
        self._scaffold_dict = {scaffold_id: max(ends[scaffold_id])-min(starts[scaffold_id])  for scaffold_id in starts}

    @property
    def scaffold_sizes(self):
        return self._scaffold_dict

    @property
    def contig_sizes(self):
        return self._contig_dict

    @property
    def contig_orientations(self) -> Dict[str, str]:
        return {contig_id: fields[2] for contig_id, fields in self._scaffold_offsets.items()}

    def map_to_scaffold_locations(self, contig_locations: LocationEntry):
        entries = [self.map_to_scaffold_location(entry) for entry in contig_locations]
        return LocationEntry.from_entry_tuples([(location.chromosome, location.position) for location in entries])

    def map_to_scaffold_location(self, contig_location: LocationEntry):
        scaffold, offset, orientation = self._scaffold_offsets[str(contig_location.chromosome)]
        assert isinstance(orientation, str)
        local_offset = int(contig_location.position)
        if orientation == '-':
            local_offset = self._contig_dict[str(contig_location.chromosome)] - int(local_offset)
        return LocationEntry.single_entry(scaffold, offset+local_offset)

    def mask_and_map_locations(self, scaffold_locations: LocationEntry):
        single_entries = (self.map_location(scaffold_location) for scaffold_location in scaffold_locations)
        return LocationEntry.from_entry_tuples([(entry.chromosome, entry.position) for entry in single_entries if entry is not None])

    def mask_and_map_location_pairs(self, scaffold_location_pairs: PairedLocationEntry):
        mapped_a = [self.map_location(scaffold_location) for scaffold_location in scaffold_location_pairs.a]
        mapped_b = [self.map_location(scaffold_location) for scaffold_location in scaffold_location_pairs.b]
        mask = [a is None or b is None for a, b in zip(mapped_a, mapped_b)]
        a = LocationEntry.from_entry_tuples([(entry.chromosome, entry.position) for entry, m in zip(mapped_a, mask) if not m])
        b = LocationEntry.from_entry_tuples([(entry.chromosome, entry.position) for entry, m in zip(mapped_b, mask) if not m])
        return PairedLocationEntry(a, b)

    def map_location(self, scaffold_location: LocationEntry):
        assert self._all_positive
        scaffold_id = str(scaffold_location.chromosome)
        scaffold_start = int(scaffold_location.position)
        start_id = np.searchsorted(self._starts[scaffold_id], scaffold_start, side='right')-1
        end_id = np.searchsorted(self._ends[scaffold_id], scaffold_start, side='right')
        if start_id != end_id:
            return None
        contig_id = self._contig_ids[scaffold_id][start_id]
        contig_start = scaffold_start - self._starts[scaffold_id][start_id]
        return LocationEntry.single_entry(contig_id, contig_start)


@bnpdataclass
class ScaffoldAlignments:
    scaffold_id: str
    scaffold_start: int
    scaffold_end: int
    contig_id: str
    contig_start: int
    contig_end: int
    orientation: str

    def to_dict(self):
        d = defaultdict(dict)
        for entry in self:
            d[str(entry.scaffold_id)][str(entry.contig_id)] = {'start': int(entry.contig_start), 'end': int(entry.contig_end), 'orientation': str(entry.orientation)}
        return d

    def to_agp(self, file_name):
        with open(file_name, "w") as f:
            counters = defaultdict(lambda: 1)
            for entry in self:
                f.write(f"{entry.scaffold_id.to_string()}\t{entry.scaffold_start + 1}\t{entry.scaffold_end + 1}\t"
                        f"{counters[entry.scaffold_id.to_string()]}\tW\t{entry.contig_id.to_string()}"
                        f"\t{entry.contig_start + 1}\t{entry.contig_end + 1}\t{entry.orientation.to_string()}\n")
                counters[entry.scaffold_id.to_string()] += 1

    @classmethod
    def from_agp(cls, file_name) -> 'ScaffoldAlignments':
        entries = []
        with open(file_name) as f:
            for line in f:
                line = line.strip().split()
                if line[-1] == 'proximity_ligation':
                    continue
                if line[6] == "scaffold":
                    continue
                entries.append(
                    (line[0],
                     int(line[1]) - 1,
                     int(line[2]) - 1,
                     line[5],
                     int(line[6]) - 1,
                     int(line[7]) - 1,
                     line[8])
                )
        return cls.from_entry_tuples(entries)

    def get_description(self):

        scaffolds = defaultdict(list)
        for entry in self:
            contig = entry.contig_id.to_string() + " " + entry.orientation.to_string()
            scaffolds[entry.scaffold_id.to_string()].append(contig)

        desc = ""
        for scaffold in scaffolds:
            desc += "Scaffold " + scaffold + ": "
            desc += ", ".join(scaffolds[scaffold])
            desc += "\n"

        return desc

    @property
    def scaffold_ids(self):
        ids = []
        for entry in self:
            scaffold_id = entry.scaffold_id.to_string()
            if scaffold_id not in ids:
                ids.append(scaffold_id)
        return ids

    def get_contigs_in_scaffold(self, scaffold_id) -> List[DirectedNode]:
        entries = self[bnp.str_equal(self.scaffold_id, scaffold_id)]
        contigs = []
        for entry in entries:
            contig_name = entry.contig_id.to_string()
            direction = entry.orientation.to_string()
            contigs.append(DirectedNode(contig_name, direction))
        return contigs

    def to_list_of_edges(self) -> List[Tuple[DirectedNode, DirectedNode]]:
        """Returns a list of edges (tuples of nodes) where edges are nodes connected because
         they are connected in a scaffold
        """
        edges = []
        for scaffold_id in self.scaffold_ids:
            contigs = self.get_contigs_in_scaffold(scaffold_id)
            if len(contigs) > 1:
                for i in range(len(contigs) - 1):
                    edges.append((contigs[i], contigs[i + 1]))
        return edges

    def get_list_of_nodes(self) -> List[List[DirectedNode]]:
        """Returns a list of lists of nodes where each sublist contains the nodes for a scaffold"""
        nodes = []
        for scaffold_id in self.scaffold_ids:
            contigs = self.get_contigs_in_scaffold(scaffold_id)
            nodes.append(contigs)
        return nodes


def translate_bam_coordinates(contig_bam_file_name: str, scaffold_alignments: ScaffoldAlignments,
                              contigs_genome: bnp.Genome, scaffolds_genome: bnp.Genome, out_file_name: str):
    """
    Reads bam and translates coordinates from contig coordinates to scaffold coordinates using
    scaffold alignments. Prints sam as output
    """
    contigs_encoding = contigs_genome.get_genome_context().encoding
    scaffolds_encoding = scaffolds_genome.get_genome_context().encoding

    logging.info(f"Contig encoding: {contigs_encoding}")

    scaffold_map = ScaffoldMap(scaffold_alignments)
    contig_sizes = contigs_genome.get_genome_context().chrom_sizes

    # contig sizes are original contigs without padding, scaffolds have padding
    # find actual size of each scaffold when contigs are consecutive
    scaffold_sizes = {}
    for scaffold_id in scaffold_map.scaffold_sizes:
        scaffold_sizes[scaffold_id] = sum(contig_sizes[contig_id] for contig_id in scaffold_map._contig_ids[scaffold_id])

    #scaffold_sizes = scaffold_map.scaffold_sizes
    #contig_sizes = scaffold_map.contig_sizes
    #scaffold_sizes = scaffolds_genome.get_genome_context().chrom_sizes
    logging.info(f"Contig sizes: {contig_sizes}")
    logging.info(f"Scaffold sizes: {scaffold_sizes}")
    assert sum(scaffold_sizes.values()) == sum(contig_sizes.values())

    scaffold_global_offset = GlobalOffset(scaffold_sizes, string_encoding=scaffolds_encoding)
    contig_global_offset = GlobalOffset(contig_sizes, string_encoding=contigs_encoding)

    is_reverse = np.zeros(len(contig_sizes), dtype=bool)
    logging.info(f"Contig orientations: {scaffold_map.contig_orientations}")
    contig_encoding_ids = contigs_encoding.encode(list(scaffold_map.contig_orientations.keys())).raw()
    contig_sizes_lookup = np.zeros(len(contig_sizes), dtype=int)
    # important to use contig sizes from genome, not from scaffold map (since they are padded)
    contig_sizes_lookup[contig_encoding_ids] = np.array([contig_sizes[contig] for contig in scaffold_map.contig_sizes.keys()])

    logging.info("Contig encoding ids: %s" % contig_encoding_ids)
    is_reverse[contig_encoding_ids] = [orientation == '-' for orientation in scaffold_map.contig_orientations.values()]
    logging.info("Is reverse: %s", is_reverse)

    out = bnp.open(out_file_name, "w")

    with bnp.open(contig_bam_file_name, lazy=False, buffer_type=bnp.SAMBuffer) as f:
        for chunk in f:
            out.write(chunk)
            continue
            positions = chunk.position
            # change positions on negative contigs
            logging.info(f"Original chromosome encoding: {chunk.chromosome.encoding}")
            contig_id = bnp.as_encoded_array(chunk.chromosome, target_encoding=contigs_encoding).raw()
            flip = is_reverse[contig_id]
            assert np.all(positions < contig_sizes_lookup[contig_id])
            positions[flip] = contig_sizes_lookup[contig_id[flip]] - positions[flip] - 1
            assert np.all(positions < contig_sizes_lookup[contig_id])
            assert np.all(positions >= 0)

            global_offset = contig_global_offset.from_local_coordinates(chunk.chromosome, positions)
            chromosome, offset = scaffold_global_offset.to_local_coordinates(global_offset)
            chromosome = bnp.change_encoding(chromosome, bnp.BaseEncoding)
            #chunk = bnp.replace(chunk, chromosome=chromosome, position=offset)
            logging.info(f"New chromosome encoding: {chunk.chromosome.encoding}")
            #print(chunk)

    out.close()
