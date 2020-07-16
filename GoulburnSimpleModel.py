import numpy as np
from numba import jit

from ControlledSystem import ControlledSystem
from DELWPdata.DataApproximations import *

# discharges_approx = ['Eildon-D', 'StuartMurrey-D', 'EastMain-D', 'Cattanach-D', 'GoulburnRiver-D']
# all_series = discharges_approx + ['Rainfall']

# all_params = {}
# for s in all_series:
#     all_params.update({s: np.load(filename_template.replace('[param]', s))})



def select_slice_by_dimension(shape, dim, slice_num):
    idx = [slice(None)] * dim + [slice_num] + [slice(None)] * (len(shape) - dim - 1)
    return tuple(idx)


def control_mask(shape, i1, i2):
    mask = np.ones(shape)
    mask[select_slice_by_dimension(shape, i1, 0)] = 0
    #mask[select_slice_by_dimension(shape, i2, -1)] = 0
    return mask



class GoulburnSimpleModel(ControlledSystem):
    '''
    Defines a system of connected dams.
    '''
    def __init__(self, n_states):
        filename_template = "D:\\Наука\\_Статьи\\__в работе\\water\\data\\_discharges_approximations\\[param]_approx.npy"

        self.RF_coeffs = np.load(filename_template.replace('[param]', 'Rainfall'))
        self.Ei_coeffs = np.load(filename_template.replace('[param]', 'Eildon-D'))
        self.SM_coeffs = np.load(filename_template.replace('[param]', 'StuartMurrey-D'))
        self.Ct_coeffs = np.load(filename_template.replace('[param]', 'Cattanach-D'))
        self.EM_coeffs = np.load(filename_template.replace('[param]', 'EastMain-D'))
        self.GR_coeffs = np.load(filename_template.replace('[param]', 'GoulburnRiver-D'))

        points = np.arange(0, 1, 0.002)
        max_discharge_Ei = np.max(approx(self.Ei_coeffs, points))
        max_discharge_SM = np.max(approx(self.SM_coeffs, points))
        max_discharge_Ct = np.max(approx(self.Ct_coeffs, points))
        max_discharge_EM = np.max(approx(self.EM_coeffs, points))
        max_discharge_GR = np.max(approx(self.GR_coeffs, points))

        to_optimize = np.array([[False, True, False, False, False, False],
                                [False, False, True, True, True, True],
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False],
                                [False, False, False, False, False, False]])
        lb = np.array([[np.NaN, 0, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, 0, 0, 0, 0],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]])
        ub = np.array([[np.NaN, max_discharge_Ei, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, max_discharge_SM, max_discharge_Ct, max_discharge_EM, max_discharge_GR],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
                       [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]])

        capacity_Eildon = 3334158  # in ML
        capacity_Goulburn = 25500

        n_states_Eildon = n_states[0]
        n_states_Goulburn = n_states[1]

        # discharge ratios: coefficient which transforms the discharge into intensity
        self.drE = n_states_Eildon / capacity_Eildon * 365
        self.drG = n_states_Goulburn / capacity_Goulburn * 365

        control_labels = ['Eildon', 'Stuart Murrey', 'Cattanach', 'East Main', 'Goulburn river']

        self.time_shift = 0.5

        super().__init__(n_states, to_optimize, lb, ub, control_labels)



    def f(self, t, U):
        _f = np.zeros(np.array(self.n_states))
        u = self.m2v(U)
        t_shifted = t + self.time_shift
        u_SM_ref = approx_scalar(self.SM_coeffs, t_shifted)
        u_Ct_ref = approx_scalar(self.Ct_coeffs, t_shifted)
        u_EM_ref = approx_scalar(self.EM_coeffs, t_shifted)
        u_GR_ref = approx_scalar(self.GR_coeffs, t_shifted)
        _f_part = (u[0] - u[1] - u[2] - u[3] - u[4]) ** 2 + \
               (u[1] - u_SM_ref) ** 2 + \
               (u[2] - u_Ct_ref) ** 2 + \
               (u[3] - u_EM_ref) ** 2 + \
               (u[4] - u_GR_ref) ** 2

        _f = _f + _f_part

        penalty = 1e5
        for i in range(0, self.n_mcs):
            for j in range(0, self.n_mcs):
                if self.to_optimize[i, j]:
                    _f = _f + penalty * U[i,j]**2 * np.abs(control_mask(self.n_states, i, j) - 1.0) # penalty for outbound control at "drought" states and inbound control at "flood" states

        return _f


    def A(self, mc_num, t, U):
        if mc_num == 0: # Eildon
            return _A_Eildon(self.n_states[mc_num], U[0,1] * self.drE, approx_scalar(self.RF_coeffs, t+self.time_shift) * self.drE)
        elif mc_num == 1: # Goulburn
            return _A_Goulburn(self.n_states[mc_num], U[0,1] * self.drG, control_out(mc_num, U, self.to_optimize) * self.drG)
        else:
            return np.zeros((1, 1))


@jit
def _A_Eildon(n, u_toGoulburn, lambda_fromRainfall):
    a = np.zeros((n, n))
    for i in range(1, n):
        a[i, i - 1] = u_toGoulburn
        a[i, i] = a[i, i] - a[i, i - 1]
        a[i - 1, i] = lambda_fromRainfall
        a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
    return a


@jit
def _A_Goulburn(n, u_toGoulburn, u_fromGoulburn):
    a = np.zeros((n, n))
    for i in range(1, n):
        a[i, i - 1] = u_fromGoulburn
        a[i, i] = a[i, i] - a[i, i - 1]
        a[i - 1, i] = u_toGoulburn
        a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
    return a


# @jit
# def _A(n, mc_num, t, U, to_optimize):
#     a = np.zeros((n, n))
#     if mc_num == 0 : # Eildon
#         for i in range(1, n):
#             a[i, i - 1] = control_out(mc_num, U, to_optimize) * drE
#             a[i, i] = a[i, i] - a[i, i - 1]
#             a[i - 1, i] = approx_scalar(RF_coeffs, t+time_shift) * drE
#             a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
#     if mc_num == 1 : # Goulburn
#         for i in range(1, n):
#             a[i, i - 1] = control_out(mc_num, U, to_optimize) * drG
#             a[i, i] = a[i, i] - a[i, i - 1]
#             a[i - 1, i] = control_in(mc_num, U, to_optimize) * drG
#             a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
#     return a



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


