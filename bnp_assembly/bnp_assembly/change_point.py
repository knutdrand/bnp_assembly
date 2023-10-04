import numpy as np
import scipy.stats


def find_change_point(data):
    sum_a = 0
    sum_b = np.sum(data)
    max_likelihood = scipy.stats.poisson(np.mean(data)).logpmf(data).sum()
    change_point = 0
    for i, value in enumerate(data):
        sum_a += value
        sum_b -= value
        mean_a = sum_a/(i+1)
        mean_b = sum_b/(len(data)-i-1)
        likelihood = scipy.stats.poisson(mean_a).logpmf(data[:i+1]).sum() + scipy.stats.poisson(mean_b).logpmf(data[i+1:]).sum()
        if likelihood > max_likelihood:
            if mean_a < mean_b//2:
                max_likelihood = likelihood
                change_point = i+1

    return change_point



