import numpy as np

rate[e, i, j] = d(i, j)

P(X[e, i, j]) = Poisson(rate[e, i, j])


class CountModel:
    def __init__(self, expected_matrix):
        self._expected_matrix = expected_matrix

    def sample(self, n):
        return np.random.poisson(self._expected_matrix, size=n)


class TruncatedCountModel:
    def __init__(self, expected_matrix):
        self._expected_matrix = expected_matrix
        self._count_model = CountModel(expected_matrix)

    def sample(self, n):
        return np.clip(self._count_model.sample(n), a_max=self._expected_matrix)
