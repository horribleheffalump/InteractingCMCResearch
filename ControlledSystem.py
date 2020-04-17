import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from BasicCMC import cronsum
from numba import jit
import os

n_states = [3, 4, 5]
np_n_states = np.array(n_states)
desirable_state = [1, 1, 1]

to_optimize = np.array([[False, True, False], [True, False, True], [False, True, False]])
lb = np.array([[np.NaN, 0, np.NaN], [0, np.NaN, 0], [np.NaN, 0, np.NaN]])
#ub = np.array([[np.NaN, 4, np.NaN], [4, np.NaN, 2], [np.NaN, 4, np.NaN]])
ub = np.array([[np.NaN, 4, np.NaN], [4, np.NaN, 4], [np.NaN, 4, np.NaN]])



def m2v(control_matrix):
    return control_matrix[np.nonzero(to_optimize)]


def v2m(control_vector):
    U = np.full_like(to_optimize, np.nan, dtype=np.double)
    U[np.nonzero(to_optimize)] = control_vector
    return U

def v2print(u):
    U = v2m(u)
    out = ''
    for i in range(0, U.shape[0]):
        for j in range(i+1, U.shape[1]):
            if not np.isnan(U[i,j]):
                out = out + f'({i+1}->{j+1}) {U[i,j]:.3f} ({j+1}->{i+1}) {U[j,i]:.3f} '
    return(out)

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

@jit
def lam(i, t):
    if i == 0:
        return 0
    elif i == 1:
        return 0
    elif i == 2:
        return 0
    else:
        return np.NaN

@jit
def mu(i, t):
    if i == 0:
        return 0
    elif i == 1:
        return 0
    elif i == 2:
        return 0
    else:
        return np.NaN

@jit
def omega(i, t):
    if i == 0:
        return 0
    elif i == 1:
        return 0
    elif i == 2:
        return 0
    else:
        return np.NaN

@jit
def A(mc_num, t, u_in, u_out):
    n = np_n_states[mc_num]
    a = np.zeros((n, n))
    for i in range(1, n):
        a[i, i - 1] = mu(mc_num, t) + omega(mc_num, t) + u_out
        a[i, i] = a[i, i] - a[i, i - 1]
        a[i - 1, i] = lam(mc_num, t) + u_in
        a[i - 1, i - 1] = a[i - 1, i - 1] - a[i - 1, i]
    return a


def select_slice_by_dimension(shape, dim, slice_num):
    idx = [slice(None)] * dim + [slice_num] + [slice(None)] * (len(shape) - dim - 1)
    return tuple(idx)


def control_mask(shape, i1, i2):
    mask = np.ones(shape)
    mask[select_slice_by_dimension(shape, i1, 0)] = 0
    mask[select_slice_by_dimension(shape, i2, -1)] = 0
    return mask

@jit
def control_out(i, U):  # i - MC number
    return np.sum(U[i, :][to_optimize[i, :]])

@jit
def control_in(i, U):  # i - MC number
    return np.sum(U[:, i][to_optimize[:, i]])


def f(t, U):
    _f = np.zeros(np.array(n_states))
    f1 = np.full_like(f, lam(0, t) - mu(0, t) - omega(0, t))
    f1 = f1 - U[0,1] * control_mask(n_states, 0, 1)
    f1 = f1 + U[1,0] * control_mask(n_states, 1, 0)
    f1 = f1**2

    f2 = np.full_like(f, lam(1, t) - mu(1, t) - omega(1, t))
    f2 = f2 + U[0,1] * control_mask(n_states, 0, 1)
    f2 = f2 - U[1,0] * control_mask(n_states, 1, 0)
    f2 = f2 + U[2,1] * control_mask(n_states, 2, 1)
    f2 = f2 - U[1,2] * control_mask(n_states, 1, 2)
    f2 = f2**2

    f3 = np.full_like(f, lam(2, t) - mu(2, t) - omega(2, t))
    f3 = f3 - U[2,1] * control_mask(n_states, 2, 1)
    f3 = f3 + U[1,2] * control_mask(n_states, 1, 2)
    f3 = f3**2

    _f = f1+f2+f3

    penalty = 1e5
    _f = _f + penalty * U[0,1]**2 * np.abs(control_mask(n_states, 0, 1)-1.0) # penalty for outbound control at "drought" states and inbound control at "flood" states
    _f = _f + penalty * U[1,0]**2 * np.abs(control_mask(n_states, 1, 0)-1.0)
    _f = _f + penalty * U[2,1]**2 * np.abs(control_mask(n_states, 2, 1)-1.0)
    _f = _f + penalty * U[1,2]**2 * np.abs(control_mask(n_states, 1, 2)-1.0)

    return _f


def terminal(n_states, desirable_state, factor):
    desirable_state = np.array(desirable_state)
    phiT = np.zeros(n_states)
    for idx, x in np.ndenumerate(phiT):
        diff = np.sum(np.abs(np.array(idx) - desirable_state))
        phiT[idx] = factor**diff
    return phiT


def rhs(t, phi, U):
    rhs = np.zeros_like(phi)
    for idx, x in np.ndenumerate(rhs):
        for i in range(0, len(idx)):
            select_slice = list(idx)
            select_slice[i] = slice(None)
            select_slice = tuple(select_slice)
            rhs[idx] = rhs[idx] + np.inner(A(i, t, control_in(i, U), control_out(i, U))[:, idx[i]], phi[select_slice])
    rhs = rhs + f(t, U)
    #rhs2 = rhs_tensor(t, phi, U)
    #print(np.abs(rhs-rhs2)<1e-10)
    return rhs


def rhs_tensor(t, phi, U):
    A_transposed = [np.transpose(A(i, t, control_in(i, U), control_out(i, U))) for i in range(0, len(n_states))]
    A_full = cronsum(A_transposed)
    rhs_tensor = np.tensordot(A_full, phi, axes=([0,2,4],[0,1,2]))
    rhs_tensor = rhs_tensor + f(t, U)
    return rhs_tensor


def rhs_p_tensor(t, p, U):
    A_transposed = [np.transpose(A(i, t, control_in(i, U), control_out(i, U))) for i in range(0, len(n_states))]
    A_full = cronsum(A_transposed)
    return np.tensordot(A_full, p, axes=([0,2,4],[0,1,2]))


def sample_path(path_num, times, controls, x0):
    # path_num is useless, it is here to comply with function signature for parallel calculation
    path = np.zeros((len(times), len(n_states)))
    path[0,] = x0
    for idt, t in enumerate(times):
        if t>0:
            previous_state = [int(i) for i in path[idt-1,]]
            control_v = controls[idt,][tuple(previous_state)]
            U = v2m(control_v)
            delta = t - times[idt-1]
            transition_probs = [(A(i, t, control_in(i, U), control_out(i, U)) * delta + np.identity(n_states[i]))[previous_state[i],:] for i in range(0, len(n_states))]
            path[idt,:] = np.array([np.random.choice(np.arange(0, n_states[i]), size=1, p = transition_probs[i])[0] for i in range(0, len(n_states))])
            #print(f'{t:.2f}', transition_probs, path[idt,:])
    return path

def average_level(p):
    average = np.zeros(len(n_states))
    for i in range(0, len(n_states)):
        level = np.zeros(tuple(n_states))
        for idx, x in np.ndenumerate(level):
            select_slices = [slice(None)] * len(idx)
            select_slices[i] = idx[i]
            select_slices = tuple(select_slices)
            level[select_slices] = idx[i]
        average[i] = np.sum(p * level)
    return average


def plot_stateandcontrol_3MCs(state, control, path, max_level=3):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.gca()
    t = [0,1,1,2,2,3]
    x = [state[0]+1, state[0]+1, state[1]+1, state[1]+1, state[2]+1, state[2]+1]
    mid_level = (max_level+1.0)/2.0
    ax.fill_between(t,x)
    tol_to_show = 1e-3
    if np.abs(control[1,0]) > tol_to_show:
        ax.arrow(1.25,mid_level,-0.5,0, color='red', head_width=0.05)
        ax.text(0.6, mid_level+0.1, f'{control[1,0]:.2f}')
    if np.abs(control[0,1]) > tol_to_show:
        ax.arrow(0.75,mid_level,0.5,0, color='red', head_width=0.05)
        ax.text(1.1, mid_level+0.1, f'{control[0,1]:.2f}')
    if np.abs(control[2,1]) > tol_to_show:
        ax.arrow(2.25,mid_level,-0.5,0, color='red', head_width=0.05)
        ax.text(1.6, mid_level+0.1, f'{control[2,1]:.2f}')
    if np.abs(control[1,2]) > tol_to_show:
        ax.arrow(1.75,mid_level,0.5,0, color='red', head_width=0.05)
        ax.text(2.1, mid_level+0.1, f'{control[1,2]:.2f}')
    ax.set_ylim(0, max_level+1)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(['1', '2', '3'])
    y_ticks = np.arange(1, max_level+1, 1)
    y_labels = ['']*len(y_ticks)
    y_labels[0] = 'min'
    y_labels[-1] = 'flood'
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    plt.savefig(f'{path}state_{state[0]}_{state[1]}_{state[2]}.png')
    plt.close(fig)

def plot_value_control(state, times, values, probabilities, controls,  path, max_level=3):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    #gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

    ax = plt.subplot(gs[0])
    ax.plot(times, probabilities, label='probability', color='red')
    ax.set_ylim(0,1)
    ax.legend(loc='upper left')
    ax_ = plt.twinx(ax)
    ax_.plot(times, values, label='value')
    ax_.legend(loc='upper right')



    ax = plt.subplot(gs[1])
    names = ['u[1,2]','u[2,1]','u[2,3]','u[3,2]']
    for k in range(0, controls.shape[1]):
        ax.plot(times, controls[:,k], label=names[k])
    ax.legend(loc='upper left')
    ax.set_ylim(np.min(lb[to_optimize])-0.5,np.max(ub[to_optimize])+0.5)
    ax.set_title(f'({state[0]}, {state[1]}, {state[2]})', y=-0.01)

    ax_ = plt.twinx(ax)
    tsize = (np.max(times) - np.min(times))/3.0
    t = [np.min(times) + tsize*x for x in [0,1,1,2,2,3]]
    x = [state[0]+1, state[0]+1, state[1]+1, state[1]+1, state[2]+1, state[2]+1]
    ax_.fill_between(t,x, alpha=0.5)
    ax_.set_axis_off()
    ax_.set_ylim(0, max_level+1)

    # tol_to_show = 1e-3
    # if np.abs(control[1,0]) > tol_to_show:
    #     ax.arrow(1.25,mid_level,-0.5,0, color='red', head_width=0.05)
    #     ax.text(0.6, mid_level+0.1, f'{control[1,0]:.2f}')
    # if np.abs(control[0,1]) > tol_to_show:
    #     ax.arrow(0.75,mid_level,0.5,0, color='red', head_width=0.05)
    #     ax.text(1.1, mid_level+0.1, f'{control[0,1]:.2f}')
    # if np.abs(control[2,1]) > tol_to_show:
    #     ax.arrow(2.25,mid_level,-0.5,0, color='red', head_width=0.05)
    #     ax.text(1.6, mid_level+0.1, f'{control[2,1]:.2f}')
    # if np.abs(control[1,2]) > tol_to_show:
    #     ax.arrow(1.75,mid_level,0.5,0, color='red', head_width=0.05)
    #     ax.text(2.1, mid_level+0.1, f'{control[1,2]:.2f}')
    # ax.set_ylim(0,4)
    # ax.set_xticks([0.5, 1.5, 2.5])
    # ax.set_xticklabels(['1', '2', '3'])
    # y_ticks = np.arange(1, max_level+1, 1)
    # y_labels = ['']*len(y_ticks)
    # y_labels[0] = 'min'
    # y_labels[-1] = 'flood'
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_labels)

    filename = f'{path}state_{state[0]}_{state[1]}_{state[2]}.png'
    plt.savefig(filename)
    plt.close(fig)


def print_slice(slice):
    for idx, control, val in slice:
        print(idx, v2print(control), val) #, cons1(res.x), cons2(res.x))


def pics_slice(slice, path):
    for idx, control, _ in slice:
        plot_stateandcontrol_3MCs(idx, v2m(control), path, max_level=np.max(n_states))


def pics_plots(times, values, probabilities, controls, path):
    for idx, _ in np.ndenumerate(np.zeros(n_states)):
        plot_value_control(idx, times, values[tuple([slice(None)]+list(idx))], probabilities[tuple([slice(None)]+list(idx))], controls[tuple([slice(None)]+list(idx)+[slice(None)])], path, max_level=np.max(n_states))


def pics_averages(times, average_levels, path):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.subplot(plt.gca())
    for i in range(0, len(n_states)):
        ax.plot(times, average_levels[:,i], label=f'MC {i}')
    ax.legend()
    plt.savefig(f'{path}average.png')
    plt.close(fig)


def save_results(values, controls, path):
    np.save(f'{path}values.npy', values)
    np.save(f'{path}controls.npy', controls)


def load_results(path):
    values = np.NaN
    controls = np.NaN
    probabilities = np.NaN
    average_levels = np.NaN
    if os.path.exists(f'{path}values.npy'):
        values = np.load(f'{path}values.npy')
    if os.path.exists(f'{path}controls.npy'):
        controls = np.load(f'{path}controls.npy')
    if os.path.exists(f'{path}probabilities.npy'):
        probabilities = np.load(f'{path}probabilities.npy')
    if os.path.exists(f'{path}average_levels.npy'):
        average_levels = np.load(f'{path}average_levels.npy')
    return values, controls, probabilities, average_levels


def slice2dphi(slice):
    dphi = np.zeros(n_states)
    for idx, _, val in slice:
        dphi[idx] = val
    return dphi


def slice2U(slice):
    U = np.zeros(n_states + list(m2v(lb).shape))
    for idx, control, _ in slice:
        U[idx] = control
    return U
