import numpy as np
from numba import jit

from ControlledSystem import ControlledSystem


def select_slice_by_dimension(shape, dim, slice_num):
    idx = [slice(None)] * dim + [slice_num] + [slice(None)] * (len(shape) - dim - 1)
    return tuple(idx)


def control_mask(shape, i1, i2):
    mask = np.ones(shape)
    mask[select_slice_by_dimension(shape, i1, 0)] = 0
    mask[select_slice_by_dimension(shape, i2, -1)] = 0
    return mask



class ControlledDams(ControlledSystem):
    '''
    Defines a system of connected dams.
    '''
    def __init__(self, n_states, controls_to_optimize, controls_lb, controls_ub, control_names):
        super().__init__(n_states, controls_to_optimize, controls_lb, controls_ub, control_names)


    def f(self, t, U):
        _f = np.zeros(np.array(self.n_states))
        for i in range(0, self.n_mcs):
            _f_i = np.full_like(_f, lam(i, t) - mu(i, t) - omega(i, t))
            for j in range(0, self.n_mcs):
                if self.to_optimize[i,j]:
                    _f_i = _f_i - U[i,j] * control_mask(self.n_states, i, j)
                if self.to_optimize[j,i]:
                    _f_i = _f_i + U[j,i] * control_mask(self.n_states, j, i)
            _f = _f + _f_i**2

        penalty = 1e5
        for i in range(0, self.n_mcs):
            for j in range(0, self.n_mcs):
                if self.to_optimize[i, j]:
                    _f = _f + penalty * U[i,j]**2 * np.abs(control_mask(self.n_states, i, j) - 1.0) # penalty for outbound control at "drought" states and inbound control at "flood" states
        return _f


    def A(self, mc_num, t, U):
        return _A(self.n_states[mc_num], mc_num, t, U, self.to_optimize)



@jit
def _A(n, mc_num, t, U, to_optimize):
    a = np.zeros((n, n))
    for i in range(1, n):
        a[i, i - 1] = mu(mc_num, t) + omega(mc_num, t) + control_out(mc_num, U, to_optimize)
        a[i, i] = a[i, i] - a[i, i - 1]
        a[i - 1, i] = lam(mc_num, t) + control_in(mc_num, U, to_optimize)
        a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
    return a

@jit
def lam(i, t):
    return 0

@jit
def mu(i, t):
    return 0
@jit
def omega(i, t):
    return 0

@jit
def control_out(i, U, to_optimize):  # i - MC number
    return np.sum(U[i, :][to_optimize[i, :]])

@jit
def control_in(i, U, to_optimize):  # i - MC number
    return np.sum(U[:, i][to_optimize[:, i]])

    # @jit
    # def lam(i, t):
    #     if i == 0:
    #         return -np.cos(2*np.pi*t) + 10
    #     elif i == 1:
    #         return -2*np.cos(2*np.pi*t) + 14
    #     elif i == 2:
    #         return -0.5*np.cos(2*np.pi*t) + 6
    #     else:
    #         return np.NaN
    #
    # @jit
    # def mu(i, t):
    #     if i == 0:
    #         return np.sin(2*np.pi*t + 5/12*np.pi) + 3
    #     elif i == 1:
    #         return 2*np.sin(2*np.pi*t + 5/12*np.pi) + 6
    #     elif i == 2:
    #         return  0.5*np.sin(2*np.pi*t + 5/12*np.pi) + 2
    #     else:
    #         return np.NaN
    #
    # @jit
    # def omega(i, t):
    #     if i == 0:
    #         return np.sin(2*np.pi*t + 1/4*np.pi) + 7
    #     elif i == 1:
    #         return 2*np.sin(2*np.pi*t + 1/4*np.pi) + 8
    #     elif i == 2:
    #         return 0.5*np.sin(2*np.pi*t + 1/4*np.pi) + 4
    #     else:
    #         return np.NaN

    # def f(self, t, U): # for 3 MCs
    #     _f = np.zeros(np.array(self.n_states))
    #     f1 = np.full_like(_f, lam(0, t) - mu(0, t) - omega(0, t))
    #     f1 = f1 - U[0,1] * control_mask(self.n_states, 0, 1)
    #     f1 = f1 + U[1,0] * control_mask(self.n_states, 1, 0)
    #     f1 = f1**2
    #
    #     f2 = np.full_like(_f, lam(1, t) - mu(1, t) - omega(1, t))
    #     f2 = f2 + U[0,1] * control_mask(self.n_states, 0, 1)
    #     f2 = f2 - U[1,0] * control_mask(self.n_states, 1, 0)
    #     f2 = f2 + U[2,1] * control_mask(self.n_states, 2, 1)
    #     f2 = f2 - U[1,2] * control_mask(self.n_states, 1, 2)
    #     f2 = f2**2
    #
    #     f3 = np.full_like(_f, lam(2, t) - mu(2, t) - omega(2, t))
    #     f3 = f3 - U[2,1] * control_mask(self.n_states, 2, 1)
    #     f3 = f3 + U[1,2] * control_mask(self.n_states, 1, 2)
    #     f3 = f3**2
    #
    #     _f = f1+f2+f3
    #
    #     penalty = 1e5
    #     _f = _f + penalty * U[0,1]**2 * np.abs(control_mask(self.n_states, 0, 1) - 1.0) # penalty for outbound control at "drought" states and inbound control at "flood" states
    #     _f = _f + penalty * U[1,0]**2 * np.abs(control_mask(self.n_states, 1, 0) - 1.0)
    #     _f = _f + penalty * U[2,1]**2 * np.abs(control_mask(self.n_states, 2, 1) - 1.0)
    #     _f = _f + penalty * U[1,2]**2 * np.abs(control_mask(self.n_states, 1, 2) - 1.0)
    #
    #     diff = np.linalg.norm(_f - self.f2(t,U))
    #     if diff > 1e-5:
    #         print(f'ACHTUNG {diff}')
    #     return _f


# def rhs(t, phi, U): # incorrect
#     rhs = np.zeros_like(phi)
#     for idx, x in np.ndenumerate(rhs):
#         for i in range(0, len(idx)):
#             select_slice = list(idx)
#             select_slice[i] = slice(None)
#             select_slice = tuple(select_slice)
#             rhs[idx] = rhs[idx] + np.inner(A(i, t, control_in(i, U), control_out(i, U))[:, idx[i]], phi[select_slice])
#     rhs = rhs + f(t, U)
#     #rhs2 = rhs_tensor(t, phi, U)
#     #print(np.abs(rhs-rhs2)<1e-10)
#     return rhs


