import typing as tp
from collections import Counter
from bnp_assembly.location import LocationPair, Location


def calculate_distance_matrices(contig_dict: tp.Dict[str, int], location_pairs: LocationPair, window_size=15):
    overlap_counts = Counter()
    inside_counts = Counter()
    for a, b in zip(location_pairs.location_a, location_pairs.location_b):
        for direction_a in ('l', 'r'):
            for direction_b in ('l', 'r'):
                if direction_a=='l':
                    offset_a = a.offset
                else:
                    offset_a = contig_dict[int(a.contig_id)]-a.offset
                if direction_b=='l':
                    offset_b = b.offset
                else:
                    offset_b = contig_dict[int(b.contig_id)]-b.offset
            id_a = (int(a.contig_id), direction_a)
            id_b = (int(b.contig_id), direction_b)
            #Case: overlap_count
            if (a.contig_id != b.contig_id) and (offset_b<window_size) and (offset_b<window_size):

                overlap_counts[frozenset((id_a, id_b))] += 1
            elif id_a == id_b:
                if min(offset_a, offset_b)<window_size and  window_size<=max(offset_a, offset_b)<2*window_size:
                    inside_counts[id_a] + 1
    return overlap_counts, inside_counts
