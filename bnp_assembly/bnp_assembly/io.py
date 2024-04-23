import itertools
import logging

import numpy as np
from bionumpy.datatypes import BamEntry
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

    if buffer is not None:
        yield buffer


def get_genomic_read_pairs(genome: bnp.Genome, bam_file_name, mapq_threshold=10) -> GenomicLocationPair:
    alignments = bnp.open(bam_file_name).read()
    mask = alignments.mapq > mapq_threshold
    interval = bnp.alignments.alignment_to_interval(alignments)
    reads = genome.get_intervals(interval)
    location = reads.get_location('stop')
    assert len(mask) % 2 == 0, "Number of reads should be even if reads are paired correctly"
    mask = mask[::2] & mask[1::2]
    return GenomicLocationPair(location[::2][mask], location[1::2][mask])


class PairedReadStream:
    """
    Represents a stream of Paired Reads. Each element in the stream is a stream of chunks of reads (LocationPair)
    """
    def __init__(self, stream: Iterable[Iterable[LocationPair]]):
        self._stream = stream

    @classmethod
    def from_pairs(cls, genome: bnp.Genome, pairs_file_name: str):
        """Creates from .pairs file"""
        pass

    @classmethod
    def from_bam(cls, genome: bnp.Genome, bam_file_name: str, mapq_threshold: int = 10):
        def get():
            chunks = bnp.open(bam_file_name).read_chunks()
            chunks = chunk_reads_into_even_number(chunks)
            for chunk in chunks:
                yield PairedReadStream.parse_bam_entry(genome, chunk, mapq_threshold)
        return cls((get() for _ in itertools.count()))

    def __iter__(self):
        return self._stream

    def __next__(self):
        return next(self._stream)

    @classmethod
    def from_location_pair(cls, location_pair: LocationPair):
        return cls(itertools.repeat([location_pair]))

    @staticmethod
    def parse_bam_entry(genome: bnp.Genome, bam_entry: BamEntry, mapq_threshold: int = 10):
        mask = bam_entry.mapq > mapq_threshold
        interval = bnp.alignments.alignment_to_interval(bam_entry)
        reads = genome.get_intervals(interval)
        location = reads.get_location('stop')
        mask = mask[::2] & mask[1::2]
        #logging.info(f"{np.sum(mask==0)/len(mask)} reads filtered out with mapq < {mapq_threshold}")
        genomic_locations = GenomicLocationPair(location[::2][mask], location[1::2][mask])
        numeric_locations = genomic_locations.get_numeric_locations()
        return numeric_locations

    @classmethod
    def from_bam_entry(cls, genome: bnp.Genome, bam_entry, mapq_threshold=10):
        locations = PairedReadStream.parse_bam_entry(genome, bam_entry, mapq_threshold)
        return cls(itertools.repeat([locations]))







