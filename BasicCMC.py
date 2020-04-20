import numpy as np


def birth_death_generator(dim, lam, mu):
    def generator(t, u_in, u_out):
        a = np.zeros((dim, dim))
        for i in range(1, dim):
            a[i, i - 1] = mu(t) + u_out
            a[i, i] = a[i, i] - a[i, i - 1]
            a[i - 1, i] = lam(t) + u_in
            a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
        return a
    return generator
