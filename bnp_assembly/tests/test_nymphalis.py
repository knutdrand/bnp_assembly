import matspy
from shared_memory_wrapper import from_file
import pytest
from bnp_assembly.cli import register_logging
from bnp_assembly.make_scaffold import path_optimization_join_and_split
import matplotlib.pyplot as plt


# outdated
@pytest.mark.skip
def test():
    register_logging("logging")
    interaction_matrix = from_file("interaction_matrix_10000_nymphalis.npz")
    interaction_matrix.plot()
    plt.show()


    interaction_matrix_clipping = from_file("interaction_matrix_1000_nymphalis.npz")
    #contig_clips = find_contig_clips_from_interaction_matrix(numeric_input_data.contig_dict, interaction_matrix_clipping, window_size=100)
    #interaction_matrix.trim_with_clips(contig_clips)

    #prob_reads_given_edge = get_prob_given_intra_background_for_edges(interaction_matrix)
    #prob_reads_given_edge.plot(name="final probs").show()
    #plt.imshow(prob_reads_given_edge.data[0::2, 1::2])
    #plt.show()

    result = path_optimization_join_and_split(interaction_matrix=interaction_matrix)

