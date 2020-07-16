import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from numba import jit

def moving_averages(x, half_window):
    x = np.append(np.array([x[0]]*half_window), x)
    x = np.append(x, np.array([x[-1]]*half_window))
    cumsum = np.cumsum(np.insert(x, 0, 0))
    window = half_window * 2
    ma = (cumsum[window:-1] - cumsum[:-window-1]) / window
    return ma

@jit
def approx(coeffs, t, period = 1.0):
    dim = int((coeffs.size - 1) / 2)
    a0 = coeffs[0]
    a = coeffs[1:dim+1]
    b = coeffs[dim+1:2*dim+1]
    tt = 2 * np.pi / period * np.arange(1, dim+1) * np.ones((t.size, dim)) * t.reshape((t.size,1))
    res = a0 + np.sum(a * np.sin(tt), axis = 1) + np.sum(b * np.cos(tt), axis = 1)
    # res = a0 \
    #       + sum(map(lambda i : a[i] * np.sin(2 * np.pi * (i+1) / period * t), range(0,dim))) \
    #       + sum(map(lambda i : b[i] * np.cos(2 * np.pi * (i+1) / period * t), range(0,dim)))
    return res

@jit
def approx_scalar(coeffs, t):
    dim = int((coeffs.size - 1) / 2)
    a0 = coeffs[0]
    a = coeffs[1:dim+1]
    b = coeffs[dim+1:2*dim+1]
    tt = 2 * np.pi * np.arange(1, dim+1) * t
    res = a0 + np.sum(a * np.sin(tt)) + np.sum(b * np.cos(tt))
    # res = a0 \
    #       + sum(map(lambda i : a[i] * np.sin(2 * np.pi * (i+1) / period * t), range(0,dim))) \
    #       + sum(map(lambda i : b[i] * np.cos(2 * np.pi * (i+1) / period * t), range(0,dim)))
    return res

class approximation():
    def __init__(self, label, points, values, ma_halfwindow, min_degree, max_degree):
        self.label = label
        self.points = points
        self.values = values
        self.ma_halfwindow = ma_halfwindow
        self.ma = moving_averages(self.values, self.ma_halfwindow)
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.coeffs = []
        self.rmses = []
        for i in range(self.min_degree, self.max_degree+1):
            p0 = np.array([1.0]*(2*i+1))
            result = optimize.leastsq(lambda p: approx(p, self.points) - self.values, p0)
            rmse = np.mean((approx(result[0], self.points) - self.values)**2)
            self.coeffs.append(np.array(result[0]))
            self.rmses.append(rmse)

    def plot_all(self):
        fig, ax1 = plt.subplots()
        ax1.set_ylabel(self.label)
        ax1.scatter(self.points, self.values, color='red', label='values', marker='.')
        ax1.plot(self.points, self.ma, color='red', label='moving average')
        for p in self.coeffs:
            ax1.plot(self.points, approx(p, self.points), label=f'approx of degree {(p.size - 1) / 2}')
        ax1.legend(loc='upper center')
        fig.tight_layout()
        plt.show()

    def plot_rmses(self):
        fig, ax1 = plt.subplots()
        ax1.plot(range(self.min_degree, self.max_degree+1), self.rmses)
        fig.tight_layout()
        plt.show()

