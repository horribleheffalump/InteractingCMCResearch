import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import gridspec

import os

from abc import ABC, abstractmethod

from functools import partial
from multiprocessing import Pool
from time import time, strftime, localtime, gmtime

# tol_constraints = 1e-10
# consts=[]
# consts.append(
#     {'type': 'ineq', 'fun': lambda u: -np.abs(u[0] * u[1]) + tol_constraints})
# consts.append(
#     {'type': 'ineq', 'fun': lambda u: -np.abs(u[2] * u[3]) + tol_constraints})

def proc(idx, t, phi, controlled_system):
    res = minimize(lambda x: controlled_system.H(t, phi, controlled_system.v2m(x))[idx], controlled_system.init, bounds=controlled_system.bounds,
                   constraints=[{'type': 'ineq', 'fun': lambda u: -np.abs(u[p[0]] * u[p[1]]) + controlled_system.tol_constraints} for p in controlled_system.incompatible_control_pairs])  # , options={'disp':True}, method='SLSQP')
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


class ControlledSystem(ABC):
    '''
    Defines a system of dependent Markov chains.
    The interaction between MC's is only through the control parameters U which are shared between the MCs
    '''
    def __init__(self, n_states, controls_to_optimize, controls_lb, controls_ub, control_names):
        '''
        Constructor
        :param n_states: list of number of states of MCs. E.g., [3,4,5] defines 3 MCs with number of states 3,4 and 5 respectively.
        :param controls_to_optimize: matrix mask of (L,L) shape, where L is the number of MCs.
        Defines which control parameters are subject to optimization.
        E.g., to_optimize = np.array([[False, True, False], [True, False, True], [False, True, False]])
        defines that U[0,1], U[1,0], U[1,2], U[2,1] are subject to optimization, while U[0,2], U[2,0] are not (i.e. there is no interaction between the 1 and 3 dam)
        :param controls_lb: lower bound for the control matrix (not necessary to define bounds for the parameters which are not subject to optimization)
        :param controls_ub: upper bound for the control matrix, e.g. np.array([[np.NaN, 4, np.NaN], [4, np.NaN, 4], [np.NaN, 4, np.NaN]])
        :param control_names: list of control names for plot legends and other output, e.g. ['U[0,1]', 'U[1,0]', 'U[1,2]', 'U[2,1]']
        '''
        self.n_states = n_states
        self.np_n_states = np.array(n_states)
        self.n_mcs = len(n_states)
        self.to_optimize = controls_to_optimize
        self.lb = controls_lb
        self.ub = controls_ub
        self.names = control_names
        self.colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

        self.values = []
        self.controls = []
        self.probs_joint_MC = []
        self.probs_joint_theor = []
        self.average_levels_theor = []
        self.average_levels_MC = []

        # control optimization parameters
        self.init = self.m2v(self.lb) # initial approximation for the iterative optimization procedure
        self.bounds = tuple([(self.m2v(self.lb)[i], self.m2v(self.ub)[i]) for i in range(0, len(self.m2v(self.lb)))]) # upper and lower control bounds in tuple form

        # define control pairs which can not be nonzero at the same time. These are all U[i,j] and U[j,i], where i != j and to_optimize[i,j]=to_optimize[j,i] = True
        self.tol_constraints = 1e-10
        temp_U = np.fromfunction(lambda i,j: i+j, (self.n_mcs, self.n_mcs))
        temp_v = self.m2v(temp_U)
        self.incompatible_control_pairs = [[i for i, x in enumerate(temp_v) if x == item] for item in set(temp_v)]

        # time mesh parameters
        self.T = np.nan
        self.delta = np.nan
        self.time_mesh = []

    @abstractmethod
    def A(self, mc_num, t, U):
        '''
        Markov chains' generators' generator
        :param mc_num: Markov chain number
        :param t: time
        :param U: control matrix
        :return: MC generator (transposed intensity matrix)
        e.g. [[-lambda, lambda, 0], [mu, -lambda-mu, lambda], [0, mu, -mu]]
        Note, that in fact this is a transposed generator,
        so to use it in the Kolmogorov equation, we have to transpose it once again
        '''
        pass

    @abstractmethod
    def f(self, t, U):
        '''
        Instant loss function for control optimization
        :param t: time
        :param U: control matrix
        :return: losses tensor f[i1,i2,....]: instant losses at time t in state [i1,i2,....]
        '''
        pass

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
        h = h + self.f(t, U)
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
            if t > 0:
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

        pool = Pool(processes=8)
        phi = self.terminal(desirable_state, 2)
        phi_enumerate = [x[0] for x in list(np.ndenumerate(phi))]
        self.values = np.zeros([self.time_mesh.shape[0]] + self.n_states)
        self.controls = np.zeros([self.time_mesh.shape[0]] + self.n_states + list(self.m2v(self.lb).shape))

        time_start = time()
        for idt, t in enumerate(self.time_mesh):
            slice = pool.map(partial(proc, t=t, phi=phi, controlled_system=self), phi_enumerate)
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
            if t>0:
                generator = np.full(np.repeat(self.n_states, 2), np.nan)
                for idp1, _ in np.ndenumerate(p_theor):
                    selection = tuple([item for sublist in ([slice(None), idp1[i]] for i in range(0,self.n_mcs)) for item in sublist])
                    #generator[idp1[0], :, idp1[1], :, idp1[2], :] = self.Generator_full(t, self.v2m(self.controls[idt,][idp1]))[idp1[0], :, idp1[1], :, idp1[2], :] # for 3 MCs
                    generator[selection] = self.Generator_full(t, self.v2m(self.controls[idt,][idp1]))[selection]
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
            if t>0:
                p_MC = np.zeros(self.n_states)
                for idp, _ in np.ndenumerate(p_MC):
                    p_MC[idp] = 1.0 - np.count_nonzero(np.linalg.norm(paths[:,idt,:]-np.array(idp), axis=1))/N
            self.probs_joint_MC[idt] = p_MC

    def plot_value_control(self, state, times, values, probabilities_theor, probabilities_MC, controls, path):
        fig = plt.figure(figsize=(6, 6), dpi=200)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        #gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

        ax = plt.subplot(gs[0])
        ax.plot(times, probabilities_theor, label='probability', color='red')
        ax.plot(times, probabilities_MC, color='red', linestyle=':')
        ax.set_ylim(0,1)
        ax.legend(loc='upper left')
        ax_ = plt.twinx(ax)
        ax_.plot(times, values, label='value') 
        ax_.legend(loc='upper right')



        ax = plt.subplot(gs[1])
        for k in range(0, controls.shape[1]):
            ax.plot(times, controls[:,k], label=self.names[k])
        ax.legend(loc='upper left')
        ax.set_ylim(np.min(self.lb[self.to_optimize])-0.5,np.max(self.ub[self.to_optimize])+0.5)
        ax.set_title(f'{state}', y=-0.01)

        ax_ = plt.twinx(ax)
        tsize = (np.max(times) - np.min(times))/self.n_mcs
        t = [np.min(times) + tsize*x for x in [val for val in range(0, self.n_mcs+1) for _ in (0, 1)][1:-1]]
        x = [val+1 for val in state for _ in (0, 1)]
        #x = [state[0]+1, state[0]+1, state[1]+1, state[1]+1, state[2]+1, state[2]+1]
        ax_.fill_between(t,x, alpha=0.5)
        ax_.set_axis_off()
        ax_.set_ylim(0, np.max(self.n_states)+1)
        filename = f'{path}state_{"_".join(str(x) for x in state)}.png'
        plt.savefig(filename)
        plt.close(fig)

    def print_slice(self, slice):
        for idx, control, val in slice:
            print(idx, self.v2print(control), val) #, cons1(res.x), cons2(res.x))

    def pics_plots(self, path):
        for idx, _ in np.ndenumerate(np.zeros(self.n_states)):
            self.plot_value_control(idx,
                                    self.time_mesh,
                                    self.values[tuple([slice(None)]+list(idx))],
                                    self.probs_joint_theor[tuple([slice(None)]+list(idx))],
                                    self.probs_joint_MC[tuple([slice(None)]+list(idx))],
                                    self.controls[tuple([slice(None)]+list(idx)+[slice(None)])],
                                    path)

    def pics_averages(self, path):
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = plt.subplot(plt.gca())
        for i in range(0, self.n_mcs):
            ax.plot(self.time_mesh, self.average_levels_theor[:, i], label=f'MC {i}', color=self.colors[i])
            ax.plot(self.time_mesh, self.average_levels_MC[:, i], linestyle=':', color=self.colors[i])
        ax.legend()
        plt.savefig(f'{path}average.png')
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
            phiT[idx] = factor ** diff
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




# def pics_slice(slice, path):
#     for idx, control, _ in slice:
#         plot_stateandcontrol_3MCs(idx, v2m(control), path, max_level=np.max(n_states))
# def plot_stateandcontrol_3MCs(state, control, path, max_level=3):
#     fig = plt.figure(figsize=(6, 6), dpi=200)
#     ax = fig.gca()
#     t = [0,1,1,2,2,3]
#     x = [state[0]+1, state[0]+1, state[1]+1, state[1]+1, state[2]+1, state[2]+1]
#     mid_level = (max_level+1.0)/2.0
#     ax.fill_between(t,x)
#     tol_to_show = 1e-3
#     if np.abs(control[1,0]) > tol_to_show:
#         ax.arrow(1.25,mid_level,-0.5,0, color='red', head_width=0.05)
#         ax.text(0.6, mid_level+0.1, f'{control[1,0]:.2f}')
#     if np.abs(control[0,1]) > tol_to_show:
#         ax.arrow(0.75,mid_level,0.5,0, color='red', head_width=0.05)
#         ax.text(1.1, mid_level+0.1, f'{control[0,1]:.2f}')
#     if np.abs(control[2,1]) > tol_to_show:
#         ax.arrow(2.25,mid_level,-0.5,0, color='red', head_width=0.05)
#         ax.text(1.6, mid_level+0.1, f'{control[2,1]:.2f}')
#     if np.abs(control[1,2]) > tol_to_show:
#         ax.arrow(1.75,mid_level,0.5,0, color='red', head_width=0.05)
#         ax.text(2.1, mid_level+0.1, f'{control[1,2]:.2f}')
#     ax.set_ylim(0, max_level+1)
#     ax.set_xticks([0.5, 1.5, 2.5])
#     ax.set_xticklabels(['1', '2', '3'])
#     y_ticks = np.arange(1, max_level+1, 1)
#     y_labels = ['']*len(y_ticks)
#     y_labels[0] = 'min'
#     y_labels[-1] = 'flood'
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(y_labels)
#     plt.savefig(f'{path}state_{state[0]}_{state[1]}_{state[2]}.png')
#     plt.close(fig)