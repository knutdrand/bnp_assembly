class ProcessReads:
    def __init__(self, read_pairs, processors):
        self._read_pairs = read_pairs
        self.proecessors = processors

        self._counts = EdgeCounts(len(self._contig_dict))

    def run(self):
        processor.setup()
        for chunk in self._read_pairs:
            for processor in self.processors:
                processor.register_location_pair(chunk)

        for processor in self.processors:
            processor.cleanup()

        for read_pair_chunk in self._read_pairs:

