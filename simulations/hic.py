import numpy as np
from scipy.stats import geom
rng  = np.random

class PairDistribution:
    def __init__(self, p, contig_length):
        self._p = p
        self._contig_length = contig_length

    def sample(self, rng, n_samples=1):
        distance = rng.geometric(self._p, size=n_samples)
        

def simulate_read(rng, contig_length, ):
    dist = geom.
    pos_1 = rng.randint(0, contig_length)
    pos_2 = 
