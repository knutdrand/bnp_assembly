import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy

def main():
    # Simulate data
    N = 100
    xlo = 0.001
    xhi = 2 * numpy.pi
    x = numpy.arange(xlo, xhi, step=(xhi - xlo) / N)
    y0 = numpy.sin(x) + numpy.log(x)
    y = y0 + numpy.random.randn(N) * 0.5

    # Prepare bases (Imat) and penalty
    z, z2 = monotonic_smooth(y)

    # Plots
    plot(x, y, z, z2)


def plot(x, y, z, z2):
    import matplotlib.pyplot as plt
    plt.scatter(x, y, linestyle='None', color='gray', s=0.5, label='raw data')
    plt.plot(x, z, color='red', label='monotonic smooth')
    plt.plot(x, z2, color='blue', linestyle='--', label='unconstrained smooth')
    plt.legend(loc="lower right")
    plt.show()


def monotonic_smooth(orig_y, subsamplig_rate=1):
    y = orig_y[::subsamplig_rate]
    dd = 3
    N = y.shape[0]
    E = numpy.eye(N)
    D3 = numpy.diff(E, n=dd, axis=0)
    D1 = numpy.diff(E, n=1, axis=0)
    la = 100
    kp = 10000000
    # Monotone smoothing
    z, z2 = _monotonic_smooth(D1, D3, E, N, kp, la, y)
    return z, z2


def _monotonic_smooth(D1, D3, E, N, kp, la, y):
    ws = numpy.zeros(N - 1)
    for it in range(30):
        Ws = numpy.diag(ws * kp)
        mon_cof = numpy.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, y)
        ws_new = (D1 @ mon_cof < 0.0) * 1
        dw = numpy.sum(ws != ws_new)
        ws = ws_new
        if (dw == 0):
            break
        print(dw)
    # Monotonic and non monotonic fits
    z = mon_cof
    #z2 = numpy.linalg.solve(E + la * D3.T @ D3, y)
    return z, None




def simulate_sklearn():
    global X, y
    n_samples = 500
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, 2)
    noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
    y = 5 * X[:, 0] + np.sin(10 * np.pi * X[:, 0]) - noise
    return X, y

#X, y = simulate_sklearn()
def smooth_sklearn(y):
    X = np.arange(y.size).reshape(-1, 1)
    gbdt_cst = HistGradientBoostingRegressor(monotonic_cst=[-1])
    gbdt_cst.fit(X, y)
    return gbdt_cst.predict(X)

