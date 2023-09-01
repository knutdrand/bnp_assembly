import numpy as np
from .location import LocationPair, Location
from .datatypes import GenomicLocationPair, StreamedGenomicLocationPair
import bionumpy as bnp
from typing import Iterable


def chunk_reads_into_even_number(read_stream):
    """
    Re-chunks reads into even number of reads
    """
    buffer = None
    size = 0
    for chunk in read_stream:
        if buffer is not None:
            size = len(buffer)
        size += len(chunk)
        if size % 2 == 1:
            new_chunk = np.concatenate([buffer, chunk[:-1]]) if buffer is not None else chunk[:-1]
            assert len(new_chunk) % 2 == 0
            buffer = chunk[-1:]
        else:
            new_chunk = np.concatenate([buffer, chunk]) if buffer is not None else chunk
        yield new_chunk

    yield buffer


def get_read_pairs(genome: bnp.Genome, bam_file_name: str, mapq_threshold=10) -> LocationPair:
    alignments = bnp.open(bam_file_name).read()
    mask = alignments.mapq > mapq_threshold
    interval = bnp.alignments.alignment_to_interval(alignments)
    reads = genome.get_intervals(interval)
    assert len(reads.data) % 2 == 0, len(reads.data)
    locations = reads.get_location('stop')
    locations = Location(locations.chromosome.raw(), locations.position)
    mask = mask[::2] & mask[1::2]
    return LocationPair(locations[::2][mask], locations[1::2][mask])


def get_genomic_read_pairs(genome: bnp.Genome, bam_file_name, mapq_threshold=10) -> GenomicLocationPair:
    alignments = bnp.open(bam_file_name).read()
    mask = alignments.mapq > mapq_threshold
    interval = bnp.alignments.alignment_to_interval(alignments)
    reads = genome.get_intervals(interval)
    location = reads.get_location('stop')
    mask = mask[::2] & mask[1::2]
    return GenomicLocationPair(location[::2][mask], location[1::2][mask])


def get_genomic_read_pairs_as_stream(genome: bnp.Genome, bam_file_name, mapq_threshold=10) -> Iterable[GenomicLocationPair]:
    alignments = bnp.open(bam_file_name).read_chunks()
    alignments = chunk_reads_into_even_number(alignments)

    def get():
        for chunk in alignments:
            mask = chunk.mapq > mapq_threshold
            interval = bnp.alignments.alignment_to_interval(chunk)
            reads = genome.get_intervals(interval)
            location = reads.get_location('stop')
            mask = mask[::2] & mask[1::2]
            yield GenomicLocationPair(location[::2][mask], location[1::2][mask])

    return StreamedGenomicLocationPair(get())