import numpy as np
import scipy.stats


def find_change_point(data):
    sum_a = 0
    sum_b = np.sum(data)
    max_likelihood = scipy.stats.poisson(np.mean(data)).logpmf(data).sum()
    change_point = 0
    cumsum = np.cumsum(data)
    total = np.sum(data)
    means_a = cumsum / np.arange(1, len(data) + 1)
    means_b = (total - cumsum) / np.arange(len(data), 0, -1)
    likelihoods = [scipy.stats.poisson(mean_a).logpmf(data[:i+1]).sum() + scipy.stats.poisson(mean_b).logpmf(data[i+1:]).sum()
                   if mean_a < mean_b // 2 else -np.inf
                   for i, (mean_a, mean_b) in enumerate(zip(means_a, means_b)) ]

    if len(likelihoods) == 0 or np.all(np.isinf(likelihoods)):
        return 0
    change_point = np.argmax(likelihoods) + 1
    # for i, value in enumerate(data):
    #     mean_a = means_a[i]
    #     mean_b = means_b[i]
    #     likelihood = scipy.stats.poisson(mean_a).logpmf(data[:i+1]).sum() + scipy.stats.poisson(mean_b).logpmf(data[i+1:]).sum()
    #     if likelihood > max_likelihood:
    #         if mean_a < mean_b//2:
    #             max_likelihood = likelihood
    #             change_point = i+1

    return change_point



