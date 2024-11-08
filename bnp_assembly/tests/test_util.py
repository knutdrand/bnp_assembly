import numpy as np


def test_logaddexp_high_precision():

    result = np.logaddexp(-0.3, -40)
