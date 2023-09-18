import numpy as np

from bnp_assembly.contig_graph import DirectedNode
import bnp_assembly.scaffolds as est
from bnp_assembly.simulation.distribution import Distribution


class Scaffolds(Distribution):
    def __init__(self, contig_dict, n_scaffolds):
        self._contig_dict = contig_dict
        self._n_scaffolds = n_scaffolds

    def sample(self):
        return self._sample()

    def _sample(self):
        scaffold_names = [f'scaffold_{i}' for i in range(self._n_scaffolds)]
        contig_dict = self._contig_dict
        if isinstance(contig_dict, Distribution):
            contig_dict = contig_dict.sample()
        contig_names = list(contig_dict)
        n_contigs = len(contig_dict)
        matching = np.random.choice(scaffold_names, size=n_contigs, replace=True)
        scaffold_list = []
        for scaffold_name in scaffold_names:
            contig_ids = [contig_name for contig_name, scaffold in zip(contig_names, matching) if scaffold == scaffold_name]
            directions = np.random.choice(['+', '-'], size=len(contig_ids))
            path = [DirectedNode(contig_id, direction) for contig_id, direction in zip(contig_ids, directions)]
            if len(path) > 0:
                scaffold_list.append(est.Scaffold(path, scaffold_name))
        return est.Scaffolds(scaffold_list)

