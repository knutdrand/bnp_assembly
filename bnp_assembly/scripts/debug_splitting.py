import pickle

import numpy as np
import typer

from bnp_assembly.scaffold_splitting.binned_bayes import BinnedBayes
import plotly.express as px


def main(matrix_filename: str, edge_info_name: str, contig_path_filename: str):
    matrix = BinnedBayes.load(matrix_filename)
    edge_info = pickle.load(open(edge_info_name, 'rb'))
    path = np.load(contig_path_filename)
    print(edge_info.keys())
    non_edges = edge_info['inter_chromsome_splits']
    print(non_edges)
    non_edges = [tuple(int(name[6:]) for name in pair) for pair in non_edges]
    edges = edge_info['intra_chromosome_splits']
    edges = set(tuple(int(name[6:]) for name in pair) for pair in edges)
    print(non_edges)
    print(matrix._contig_start_stops)
    table = matrix.triangle_table()
    mask = np.array([((path[m], path[m+1]) in edges) or ((path[m+1], path[m]) in edges) for m in table.m])
    # first_nodes = {pair[0] for pair in non_edges}
    # mask = [m in first_nodes for m in table.m]
    print(table[~mask])
    table['edge'] = mask
    return table



if __name__ == '__main__':
    # typer.run(main)
    result = main(
        '../../hic-assembly-benchmarking/matrix.npz',
        '../../hic-assembly-benchmarking/data/sacCer3/simulated/big/10/100000/not_assembled/10/hifiasm.hic.p_ctg.fa.edge_info',
        '../../hic-assembly-benchmarking/contig_path.npy')
