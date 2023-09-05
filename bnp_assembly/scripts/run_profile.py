from bnp_assembly.cli import scaffold


def test():
    dir = "../../benchmarking/"
    scaffold(
        contig_file_name=dir+"data/sacCer3/simulated/small/10/40000/123/not_assembled/20/0/hifiasm.hic.p_ctg.fa",
        read_filename=dir+"data/sacCer3/simulated/small/10/40000/123/not_assembled/20/0/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
        out_file_name=dir+"data/sacCer3/simulated/small/10/40000/123/not_assembled/20/0/bnp_scaffolding/scaffolds.fa",
        threshold=0.001,
        logging_folder=dir+"data/sacCer3/simulated/small/10/40000/123/not_assembled/20/0/bnp_scaffolding/logging",
        bin_size=3568
    )
