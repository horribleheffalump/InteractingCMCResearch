from numba import jit
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import gridspec

import os

from functools import partial
from multiprocessing import Pool
from time import time, strftime, localtime, gmtime

from matplotlib import rc

rc('font',**{'family':'serif'})
rc('text', usetex=True)
#rc('text.latex',unicode=True)

# tol_constraints = 1e-10
# consts=[]
# consts.append(
#     {'type': 'ineq', 'fun': lambda u: -np.abs(u[0] * u[1]) + tol_constraints})
# consts.append(
#     {'type': 'ineq', 'fun': lambda u: -np.abs(u[2] * u[3]) + tol_constraints})
from DELWPdata.DataApproximations import moving_averages


def proc(idx, t, phi, controlled_system):
    # res = minimize(lambda x: controlled_system.H(t, phi, controlled_system.v2m(x))[idx], controlled_system.init,
    #                jac=lambda x: np.array([controlled_system.H(t, phi, controlled_system.v2m(np.array([1.0, 0.0, 0.0])))[idx],
    #                              controlled_system.H(t, phi, controlled_system.v2m(np.array([0.0, 1.0, 0.0])))[idx],
    #                              controlled_system.H(t, phi, controlled_system.v2m(np.array([0.0, 0.0, 1.0])))[idx]]),
    #                bounds=controlled_system.bounds,
    #                #options={'disp':True}
    #                )  # , options={'disp':True}, method='SLSQP')
    res = minimize(lambda x: controlled_system.H_indexed(t, phi, controlled_system.v2m(x), idx), controlled_system.init,
                   bounds=controlled_system.bounds,
                   #options={'disp':True}
                   )  # , options={'disp':True}, method='SLSQP')

    return idx, res.x, res.fun



def cronsum(A):
    '''
    Calculates Kronecker sum of square matrices from the list A
    :param A: List of square matrices
    :return: Kronecker sum in tensor form
    '''
    #A - list of matrices
    res = A[0]
    Eres = np.eye(A[0].shape[0])
    for i in range(1, len(A)):
        Ei = np.eye(A[i].shape[0])
        res = np.multiply.outer(res, Ei) + np.multiply.outer(Eres, A[i])
        Eres = np.multiply.outer(Eres, Ei)
    return res


def select_slice_by_dimension(shape, dim, slice_num):
    idx = [slice(None)] * dim + [slice_num] + [slice(None)] * (len(shape) - dim - 1)
    return tuple(idx)


def control_mask(shape, i1, i2):
    mask = np.ones(shape)
    mask[select_slice_by_dimension(shape, i1, 0)] = 0
    mask[select_slice_by_dimension(shape, i2, -1)] = 0
    return mask


class ControlledDamsSimplified():
    '''
    Defines a system of connected dams.
    '''

    def __init__(self, n_states, controls_to_optimize, controls_lb, controls_ub, mc_names, control_names):
        '''
        Constructor
        :param n_states: list of number of states of MCs. E.g., [3,4,5] defines 3 MCs with number of states 3,4 and 5 respectively.
        :param controls_to_optimize: matrix mask of (L,L) shape, where L is the number of MCs.
        Defines which control parameters are subject to optimization.
        E.g., to_optimize = np.array([[False, True, False], [True, False, True], [False, True, False]])
        defines that U[0,1], U[1,0], U[1,2], U[2,1] are subject to optimization, while U[0,2], U[2,0] are not (i.e. there is no interaction between the 1 and 3 dam)
        :param controls_lb: lower bound for the control matrix (not necessary to define bounds for the parameters which are not subject to optimization)
        :param controls_ub: upper bound for the control matrix, e.g. np.array([[np.NaN, 4, np.NaN], [4, np.NaN, 4], [np.NaN, 4, np.NaN]])
        :param mc_names: list of Markov chain names for plot legends and other output, e.g. ['Dam 1', 'Dam 2', 'Dam 3']
        :param control_names: list of control names for plot legends and other output, e.g. ['U[0,1]', 'U[1,0]', 'U[1,2]', 'U[2,1]']
        '''
        self.n_states = n_states
        self.np_n_states = np.array(n_states)
        self.n_mcs = len(n_states)
        self.to_optimize = controls_to_optimize
        self.lb = controls_lb
        self.ub = controls_ub

        self.values = []
        self.controls = []
        self.probs_joint_MC = []
        self.probs_joint_theor = []
        self.average_levels_theor = []
        self.average_levels_MC = []

        # control optimization parameters
        self.init = self.m2v(self.lb)   # initial approximation for the iterative optimization procedure
        self.bounds = tuple([(self.m2v(self.lb)[i], self.m2v(self.ub)[i]) for i in
                             range(0, len(self.m2v(self.lb)))])  # upper and lower control bounds in tuple form

        # time mesh parameters
        self.T = np.nan
        self.delta = np.nan
        self.time_mesh = []

        # plotting parameters
        self.mc_names = mc_names
        self.control_names = control_names
        self.colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'red', 'navy']

        self.h_calls = 0

    def H_indexed(self, t, phi, U, index):
        '''
        Element [index] of the hamiltonian of the dynamic programing equation in tensor form <d(phi), X>/dt = - min_u H(t, phi, U, X)
        :param t: time
        :param phi: tensor phi[i1,i2,....] of value function values at time t in state [i1,i2,....]
        :param U: control matrix
        :param index: index of Hamiltonian element to return
        :return: Hamiltonian in tensor form H[i1,i2,....]
        '''
        #A_full = self.Generator_full(t, U)
        #h = np.tensordot(A_full, phi, axes=(list(range(0, 2 * self.n_mcs, 2)), list(range(0, self.n_mcs)))) # axes, e.g., [0,2,4][0,1,2] for 3 MC case (the same as [1,3,5][0,1,2] for the generator constructed from the transposed matrices)

        A_index = np.zeros_like(phi)
        for i in range(0, len(self.n_states)):
            A_i = np.zeros_like(phi)
            A_i_index = list(index)
            A_i_index[i] = slice(None)
            A_i[tuple(A_i_index)] = self.A(i, t, U)[index[i],:]
            A_index = A_index + A_i
        h_index = np.sum(phi*A_index)

        #print(h_index - h[index])

        return h_index


    def H(self, t, phi, U):
        '''
        Hamiltonian of the dynamic programing equation in tensor form <d(phi), X>/dt = - min_u H(t, phi, U, X)
        :param t: time
        :param phi: tensor phi[i1,i2,....] of value function values at time t in state [i1,i2,....]
        :param U: control matrix
        :return: Hamiltonian in tensor form H[i1,i2,....]
        '''
        # A_full = self.Generator_transposed(t, U)
        # h = np.tensordot(A_full, phi, axes=(list(range(0, 2 * self.n_mcs, 2)), list(range(0, self.n_mcs)))) # axes, e.g., [0,2,4][0,1,2] for 3 MC case
        A_full = self.Generator_full(t, U)
        h = np.tensordot(A_full, phi, axes=(list(range(0, 2 * self.n_mcs, 2)), list(range(0, self.n_mcs)))) # axes, e.g., [0,2,4][0,1,2] for 3 MC case (the same as [1,3,5][0,1,2] for the generator constructed from the transposed matrices)
        #h = h + self.f(t, U)
        self.h_calls = self.h_calls + 1
        return h


    def Generator_full(self, t, U):
        '''
        Compound MC generator in tensor form, which is Kronecker sum of the MCs' generators
        :param t: time
        :param U: control matrix
        :return: generator in tensor form
        '''
        # transpose the local generators, since they are defined transposed!!!
        A_full = [np.transpose((self.A(i, t, U))) for i in range(0, self.n_mcs)]
        A_full = cronsum(A_full)
        return A_full

    def sample_path(self, path_num, times, controls, x0):
        '''
        Generates a sample path of the compound MC given the controls and initial condition
        :param path_num: path number (useless, it is here to comply with function signature for parallel calculation)
        :param times: time mesh
        :param controls: controls defined on the time mesh
        :param x0: initial condition
        :return: sample path X(t) = X[t, i1, i2, ....]
        '''
        # path_num is
        path = np.zeros((len(times), self.n_mcs))
        path[0, ] = x0
        for idt, t in enumerate(times):
            if idt > 0:
                previous_state = [int(i) for i in path[idt-1,]]
                control_v = controls[idt,][tuple(previous_state)]
                U = self.v2m(control_v)
                delta = t - times[idt-1]
                transition_probs = [(self.A(i, t, U) * delta + np.identity(self.n_states[i]))[previous_state[i], :] for i in range(0, self.n_mcs)]
                path[idt, :] = np.array([np.random.choice(np.arange(0, self.n_states[i]), size=1, p=transition_probs[i])[0] for i in range(0, self.n_mcs)])
        return path


    def average_level(self, p):
        '''
        Calculates average level given the probability of states
        :param p: probabilities of compound states
        :return: average levels
        '''
        average = np.zeros(self.n_mcs)
        for i in range(0, self.n_mcs):
            level = np.zeros(tuple(self.n_states))
            for idx, x in np.ndenumerate(level):
                select_slices = [slice(None)] * len(idx)
                select_slices[i] = idx[i]
                select_slices = tuple(select_slices)
                level[select_slices] = idx[i]
            average[i] = np.sum(p * level)
        return average

    def m2v(self, control_matrix):
        '''
        Matrix to vector control transformation
        :param control_matrix: control in matrix form
        :return: control in vector form
        '''
        return control_matrix[np.nonzero(self.to_optimize)]

    def v2m(self, control_vector):
        '''
        Vector to matrix control transformation
        :param control_vector: control in vector form
        :return: control in matrix form
        '''
        U = np.full_like(self.to_optimize, np.nan, dtype=np.double)
        U[np.nonzero(self.to_optimize)] = control_vector
        return U

    def v2print(self, u):
        '''
        Print control
        :param u: control in vector form to print
        :return: string output
        '''
        U = self.v2m(u)
        out = ''
        for i in range(0, U.shape[0]):
            for j in range(i+1, U.shape[1]):
                if not np.isnan(U[i,j]):
                    out = out + f'({i+1}->{j+1}) {U[i,j]:.3f} ({j+1}->{i+1}) {U[j,i]:.3f} '
        return(out)

    def optimize_controls(self, desirable_state, T, delta, show_output=False):
        '''
        Finds an optimal control as a solution to the dynamic programming equation
        :param desirable_state: the desirable system state at the end of the simulation interval T
        :param T: end of simulation interval
        :param delta: time mesh step
        :param show_output: id True, prints detailed output of the optimization procedure
        :return:
        '''
        self.T = T
        self.delta = delta
        self.time_mesh = np.arange(T, 0.0 - delta / 2, -delta) # backward time_mesh

        pool = Pool(processes=7)
        phi = self.terminal(desirable_state, 2)
        phi_enumerate = [x[0] for x in list(np.ndenumerate(phi))]
        self.values = np.zeros([self.time_mesh.shape[0]] + self.n_states)
        self.controls = np.zeros([self.time_mesh.shape[0]] + self.n_states + list(self.m2v(self.lb).shape))

        time_start = time()
        for idt, t in enumerate(self.time_mesh):
            slice = pool.map(partial(proc, t=t, phi=phi, controlled_system=self), phi_enumerate)
            # slice = map(partial(proc, t=t, phi=phi, controlled_system=self), phi_enumerate)

            # def proc(idx):
            #     res = minimize(lambda x: self.H(t, phi, self.v2m(x))[idx], self.init,
            #              jac=lambda x: np.array(
            #                  [self.H(t, phi, self.v2m(np.array([1.0, 0.0, 0.0])))[idx],
            #                   self.H(t, phi, self.v2m(np.array([0.0, 1.0, 0.0])))[idx],
            #                   self.H(t, phi, self.v2m(np.array([0.0, 0.0, 1.0])))[idx]]),
            #              bounds=self.bounds,
            #              # options={'disp':True}
            #              )
            #     return idx, res.x, res.fun

            #slice = map(proc, phi_enumerate)
            print(self.h_calls)

            # , options={'disp':True}, method='SLSQP')

            phi = phi + delta * self.slice2dphi(slice)
            time_elapsed = time() - time_start
            time_step_average = time_elapsed / (idt + 1)
            time_rest = time_step_average * len(self.time_mesh) - time_elapsed
            time_finish = time() + time_rest
            print(f'step: {t}, elapsed: {strftime("%H:%M:%S", gmtime(time_elapsed))}, average step: {time_step_average:.3f} sec, approximate finish time: {strftime("%d %b %Y %H:%M:%S", localtime(time_finish))} ({strftime("%H:%M:%S", gmtime(time_rest))} rest)')
            if show_output:
                self.print_slice(slice)
            self.values[len(self.time_mesh)-idt-1,] = phi
            self.controls[len(self.time_mesh)-idt-1,] = self.slice2U(slice)

        self.time_mesh = np.flip(self.time_mesh) # reverse the time mesh

    def calculate_probabilities_theor(self, start_state):
        '''
        Calculates the state probabilities as a solution to the Kolmogorov equation. Also calculates the average MC levels (state number corresponds to the level)
        :param start_state: initial system state
        :return:
        '''
        self.probs_joint_theor = np.zeros([self.time_mesh.shape[0]] + self.n_states)
        self.average_levels_theor = np.zeros([self.time_mesh.shape[0]] + [self.n_mcs])

        p_theor = np.zeros(self.n_states)
        p_theor[tuple(start_state)] = 1.0

        for idt, t in enumerate(self.time_mesh):
            if idt > 0:
                generator = np.full(np.repeat(self.n_states, 2), np.nan)
                for idp1, _ in np.ndenumerate(p_theor):
                    selection = tuple([item for sublist in ([slice(None), idp1[i]] for i in range(0,self.n_mcs)) for item in sublist])
                    #generator[idp1[0], :, idp1[1], :, idp1[2], :] = self.Generator_full(t, self.v2m(self.controls[idt,][idp1]))[idp1[0], :, idp1[1], :, idp1[2], :] # for 3 MCs
                    # generator calculation dumb but fast
                    A_index = np.zeros(self.n_states)
                    for i in range(0, len(self.n_states)):
                        A_i = np.zeros(self.n_states)
                        A_i_index = list(idp1)
                        A_i_index[i] = slice(None)
                        A_i[tuple(A_i_index)] = self.A(i, t, self.v2m(self.controls[idt,][idp1]))[idp1[i], :]
                        A_index = A_index + A_i
                    generator[selection] = A_index
                    # generator calculation smart but slow
                    # generator[selection] = self.Generator_full(t, self.v2m(self.controls[idt,][idp1]))[selection]
                p_theor = p_theor + (t - self.time_mesh[idt-1]) * np.tensordot(generator, p_theor, axes=(list(range(1, 2 * self.n_mcs + 1, 2)), list(range(0, self.n_mcs)))) # axes, e.g., [1,3,5][0,1,2] for 3 MC case
            self.probs_joint_theor[idt] = p_theor
            self.average_levels_theor[idt,] = self.average_level(p_theor)

    def calculate_probabilities_MC(self, start_state):
        '''
        Calculates the state probabilities as a result of the Monte Carlo sampling. Also calculates the average MC levels (state number corresponds to the level)
        :param start_state: initial system state
        :return:
        '''
        N = 100
        pool = Pool(processes=8)
        paths = pool.map(partial(self.sample_path, times=self.time_mesh, controls=self.controls, x0=start_state), range(0, N))
        paths = np.array(paths)

        self.probs_joint_MC = np.zeros([self.time_mesh.shape[0]] + self.n_states)
        self.average_levels_MC = np.average(np.array(paths), axis=0)

        p_theor = np.zeros(self.n_states)
        p_theor[tuple(start_state)] = 1.0
        p_MC = p_theor

        for idt, t in enumerate(self.time_mesh):
            if idt > 0:
                p_MC = np.zeros(self.n_states)
                for idp, _ in np.ndenumerate(p_MC):
                    p_MC[idp] = 1.0 - np.count_nonzero(np.linalg.norm(paths[:,idt,:]-np.array(idp), axis=1))/N
            self.probs_joint_MC[idt] = p_MC

    def plot_value_control(self, state, times, values, probabilities_theor, probabilities_MC, controls, path, legend_pos, x_ticks, x_labels, pic_type):
        labels_toplot = [s for s in self.mc_names if s != '']
        mc_count_toplot = len(labels_toplot)
        n_states_toplot = [i for i in self.n_states if i > 1]

        fig = plt.figure(figsize=(10, 6), dpi=150)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[4, 1])
        #gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

        ax = plt.subplot(gs[0,0])
        ax.plot(times, probabilities_theor, label='probability', color='red')
        ax.plot(times, probabilities_MC, color='red', linestyle=':')
        ax.set_ylim(0,1)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Probability of state')

        #ax.legend(loc='upper left')
        ax_ = plt.twinx(ax)
        ax_.plot(times, values, label='value', color='navy')
        ax_.set_ylabel('Value function')
        #ax_.legend(loc='upper right')

        ax = plt.subplot(gs[0,1]) #labels upper plot
        ax.plot([], [], label='Probability', color='red')
        ax.plot([], [], label='Value function', color='navy')
        ax.legend(loc='upper left')
        ax.set_axis_off()

        ax = plt.subplot(gs[1,0])
        for k in range(0, controls.shape[1]):
            ax.plot(times, moving_averages(controls[:,k], 10), label=self.control_names[k])
        #ax.legend(loc='upper left')
        c_min = np.min(self.lb[self.to_optimize])
        c_max = np.max(self.ub[self.to_optimize])
        diff = c_max-c_min
        ax.set_ylim(c_min-0.05*diff, c_max+0.05*diff)
        if (len(x_ticks)> 0):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)

        ax.set_ylabel('Control values')

        title = 'States: ' + ', '.join([f'{t[0]} - {t[1]+1}' for t in zip(labels_toplot, state)])
        #ax.set_title(title, y=-0.15)
        #ax.legend(loc=legend_pos)

        ax_ = plt.twinx(ax)
        # tsize = (np.max(times) - np.min(times))/self.n_mcs
        # t = [np.min(times) + tsize*x for x in [val for val in range(0, self.n_mcs+1) for _ in (0, 1)][1:-1]]
        # x = [val+1 for val in state for _ in (0, 1)]
        # ax_.fill_between(t,x, alpha=0.5)

        for i in range(0, self.n_mcs):
            if (self.mc_names[i] != ''):
                bar_heights = np.zeros_like(n_states_toplot)
                bar_heights[i] = self.n_states[i]
                ax_.bar(np.arange(0, 1, 1.0 / mc_count_toplot) + 0.5 / mc_count_toplot, bar_heights, color='white', edgecolor=self.colors[i],
                       width=0.7 / mc_count_toplot, alpha=0.3)
                bar_heights[i] = state[i]+1
                ax_.bar(np.arange(0, 1, 1.0 / mc_count_toplot) + 0.5 / mc_count_toplot, bar_heights, color=self.colors[i],
                       width=0.7 / mc_count_toplot, alpha=0.3)

        #ax_.set_axis_off()
        ax_.set_ylabel('State number')
        ax_.set_ylim(0, np.max(self.n_states)+1)

        ax = plt.subplot(gs[1,1]) #labels lower plot
        for k in range(0, controls.shape[1]):
            ax.plot([], [], label=self.control_names[k])
        ax.legend(loc='lower left')
        ax.set_axis_off()

        filename = f'{path}state_{"_".join(str(x) for x in state)}.{pic_type}'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def print_slice(self, slice):
        for idx, control, val in slice:
            print(idx, self.v2print(control), val) #, cons1(res.x), cons2(res.x))

    def pics_plots(self, path, legend_pos='lower right', x_ticks=[], x_labels=[], pic_type='pdf'):
        for idx, _ in np.ndenumerate(np.zeros(self.n_states)):
            self.plot_value_control(idx,
                                    self.time_mesh,
                                    self.values[tuple([slice(None)]+list(idx))],
                                    self.probs_joint_theor[tuple([slice(None)]+list(idx))],
                                    self.probs_joint_MC[tuple([slice(None)]+list(idx))],
                                    self.controls[tuple([slice(None)]+list(idx)+[slice(None)])],
                                    path,
                                    legend_pos, x_ticks, x_labels, pic_type)

    def pics_averages(self, path, legend_pos='lower right', x_ticks=[], x_labels=[], pic_type='pdf'):
        mc_count_toplot = len([s for s in self.mc_names if s != ''])
        n_states_toplot = [i for i in self.n_states if i > 1]
        fig = plt.figure(figsize=(10, 3), dpi=150)
        #ax = plt.subplot(plt.gca())
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

        ax = plt.subplot(gs[0, 0])
        for i in range(0, self.n_mcs):
            if (self.mc_names[i] != ''):
                ax.plot(self.time_mesh, 1 + self.average_levels_theor[:, i], label=self.mc_names[i], color=self.colors[i])
                ax.plot(self.time_mesh, 1 + self.average_levels_MC[:, i], linestyle=':', color=self.colors[i])
                bar_heights = np.zeros_like(n_states_toplot)
                bar_heights[i] = self.n_states[i]
                ax.bar(np.arange(0, 1, 1.0/mc_count_toplot) + 0.5/mc_count_toplot, bar_heights, color=self.colors[i], width=0.7/mc_count_toplot, alpha=0.3)

        ax.set_ylim(0, np.max(self.n_states)+1)
        if (len(x_ticks)> 0):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
        #ax.legend(loc=legend_pos)
        ax.set_ylabel('Average state')

        ax = plt.subplot(gs[0, 1])
        for i in range(0, self.n_mcs):
            if (self.mc_names[i] != ''):
                ax.plot([], [], label=self.mc_names[i], color=self.colors[i])
        ax.set_axis_off()
        ax.legend(loc='lower left')

        plt.tight_layout()
        plt.savefig(f'{path}average.{pic_type}')
        plt.close(fig)

    def slice2dphi(self, slice):
        dphi = np.zeros(self.n_states)
        for idx, _, val in slice:
            dphi[idx] = val
        return dphi


    def slice2U(self, slice):
        U = np.zeros(self.n_states + list(self.m2v(self.lb).shape))
        for idx, control, _ in slice:
            U[idx] = control
        return U

    def terminal(self, desirable_state, factor):
        desirable_state = np.array(desirable_state)
        phiT = np.zeros(self.n_states)
        for idx, x in np.ndenumerate(phiT):
            diff = np.sum(np.abs(np.array(idx) - desirable_state))
            phiT[idx] = factor ** diff * 1e6
        return phiT

    def load_results(self, path):
        if os.path.exists(f'{path}time_mesh.npy'):
            self.time_mesh = np.load(f'{path}time_mesh.npy')
        if os.path.exists(f'{path}values.npy'):
            self.values = np.load(f'{path}values.npy')
        if os.path.exists(f'{path}controls.npy'):
            self.controls = np.load(f'{path}controls.npy')
        if os.path.exists(f'{path}probabilities_theor.npy'):
            self.probs_joint_theor = np.load(f'{path}probabilities_theor.npy')
        if os.path.exists(f'{path}probabilities_MC.npy'):
            self.probs_joint_MC = np.load(f'{path}probabilities_MC.npy')
        if os.path.exists(f'{path}average_levels_theor.npy'):
            self.average_levels_theor = np.load(f'{path}average_levels_theor.npy')
        if os.path.exists(f'{path}average_levels_MC.npy'):
            self.average_levels_MC = np.load(f'{path}average_levels_MC.npy')





    def f(self, t, U):
        _f = np.zeros(np.array(self.n_states))
        # for i in range(0, self.n_mcs):
        #     _f_i = np.full_like(_f, lam(i, t) - mu(i, t) - omega(i, t))
        #     for j in range(0, self.n_mcs):
        #         if self.to_optimize[i,j]:
        #             _f_i = _f_i - U[i,j] * control_mask(self.n_states, i, j)
        #         if self.to_optimize[j,i]:
        #             _f_i = _f_i + U[j,i] * control_mask(self.n_states, j, i)
        #     _f = _f + _f_i**2

        penalty = 1e5
        for i in range(0, self.n_mcs):
            for j in range(0, self.n_mcs):
                if self.to_optimize[i, j]:
                    _f = _f + penalty * U[i,j]**2 * np.abs(control_mask(self.n_states, i, j) - 1.0) # penalty for outbound control at "drought" states and inbound control at "flood" states
        return _f


    def A(self, mc_num, t, U):
        a = np.zeros((self.n_states[mc_num], self.n_states[mc_num]))
        for i in range(1, self.n_states[mc_num]):
            if mc_num < U.shape[0] - 1:
                a[i, i - 1] = U[mc_num, mc_num + 1]
                a[i, i] = a[i, i] - a[i, i - 1]
            if mc_num > 0:
                a[i - 1, i] = U[mc_num - 1, mc_num]
                a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
        return a




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


