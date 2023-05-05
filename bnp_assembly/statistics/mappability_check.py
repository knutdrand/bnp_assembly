import plotly.express as px
import numpy as np


def simulate(distance_distribution: np.ndarray,  size: int, n_samples: int):
    assert distance_distribution[0] == 0
    pad = distance_distribution.size
    position_1 = np.random.randint(-pad, size+pad, n_samples)

    distance = np.random.choice(np.arange(-pad+1, pad), 
                                size=n_samples, 
                                p = np.concatenate((distance_distribution[-1:0:-1], distance_distribution))/2)
    position_2 = position_1 + distance
    print(np.mean(position_1), np.mean(position_2))

    mask_2 = (position_2 >= 0) & (position_2<size)
    mask_1 = (position_1 >= 0) & (position_1<size)
    n_pairs = np.sum(mask_1 & mask_2)
    positions = np.concatenate([position_1[mask_1], position_2[mask_2]])
    return positions, n_pairs


def estimator(distance_distribution: np.ndarray, size: int, samples: np.ndarray):
    cumulative_distribution = np.cumsum(distance_distribution)/2
    position_distribution = cumulative_distribution[np.arange(size)] + cumulative_distribution[np.arange(size)[::-1]]
    print(position_distribution)
    ps = position_distribution[samples]
    print(len(samples), len(ps))
    print(np.mean(ps), np.sum(ps))
    return np.sum(ps)/2
    # for sample in samples:
    #     pre_p = cumulative_distribution[sample]
    #     post_p = cumulative_distribution[size-sample-1]
    #     ps.append(pre_p+post_p)
    #     #print(ps)
    # return np.sum(ps)/2
    

w = np.arange(21)[::-1]
w[0] = 0
p = w/np.sum(w)


true, estimated = ([], [])
for i in range(1000):
    samples, n_pairs = simulate(p, 20, 10000)
    print(len(samples), n_pairs)
    e = estimator(p, 20, samples)
    true.append(n_pairs)
    estimated.append(e)
print(estimated)
px.histogram(e-true).show()
px.histogram(e/true).show()
# scatter(x=true, y=estimated).show()
