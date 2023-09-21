import bionumpy as bnp
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


def check_bam_file(bam_filename, fasta_filename):
    locations_a, locations_b = get_location_pair(bam_filename, fasta_filename)
    mask = locations_a.chromosome == locations_b.chromosome
    locations_a = locations_a[mask]
    locations_b = locations_b[mask]
    distances = np.abs(locations_a.position - locations_b.position)
    return distances, locations_a.flag


def get_location_pair(bam_filename, fasta_filename):
    genome = bnp.Genome.from_file(fasta_filename)
    bam = bnp.open(bam_filename).read_chunk(min_chunk_size=50000000)
    locations = genome.get_locations(bam).data
    locations_b = locations[1::2]
    locations_a = locations[0::2][:len(locations_b)]
    return locations_a, locations_b


def debug_01(locations_a, locations_b, chrom_size_a):
    mask_01 = (locations_a.chromosome.raw()==0) & (locations_b.chromosome.raw() == 1)
    mask_10 = (locations_a.chromosome.raw()==1) & (locations_b.chromosome.raw() == 0)
    dist_01 = chrom_size_a - locations_a.position[mask_01] + locations_b.position[mask_01]
    dist_10 = chrom_size_a - locations_b.position[mask_10] + locations_a.position[mask_10]
    return np.concatenate([dist_01, dist_10])

folder_name = '../../benchmarking/data/athalia_rosea/real/big/10/10000000/1/not_assembled/0/200/0.0/0/'
bam_file_name = folder_name+'hifiasm.hic.p_ctg.sorted_by_read_name.bam'
fasta_file_name = folder_name + 'hifiasm.hic.p_ctg.fa'
locations_a, locations_b = get_location_pair(bam_file_name, fasta_file_name)
chrom_size = bnp.Genome.from_file(fasta_file_name).get_genome_context().chrom_sizes['contig0']
ab_distances = debug_01(locations_a, locations_b, chrom_size)
plt.hist(ab_distances, bins=100);plt.show()


distances, flag = check_bam_file(bam_file_name, fasta_file_name)


83, 147, 67,
plt.style.use("seaborn")
#plt.plot(flag, distances, '.')
# print(np.unique(flag))
t = lambda x: x
for flag_value in [65,81, 97, 113]:
    local_distances = distances[flag == flag_value]
    plt.hist(t(local_distances), bins=100)
    random_data = np.random.geometric(1/np.mean(local_distances), size=local_distances.size)
    #plt.hist(t(random_data), bins=100)
    plt.title(str(flag_value))
    plt.show()
#random_data = np.random.geometric(0.0001, size=100000)
#plt.hist(np.log(random_data+1), bins=100)
#plt.show()
# plt.hist(np.log(distances+1), bins=100)
# plt.title('log')
# plt.show()
# plt.title('not log')
# plt.hist(distances, bins=100)
# plt.show()

#px.histogram(distances).show()

