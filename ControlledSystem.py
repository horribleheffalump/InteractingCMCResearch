import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from BasicCMC import birth_death_generator

n_states = [4,4,4]

to_optimize = np.array([[False, True, False], [True, False, True], [False, True, False]])
lb = np.array([[np.NaN, 0, np.NaN], [0, np.NaN, 0], [np.NaN, 0, np.NaN]])
ub = np.array([[np.NaN, 4, np.NaN], [4, np.NaN, 2], [np.NaN, 4, np.NaN]])




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

def lambda_1(t): return -np.cos(2*np.pi*t) + 10
def lambda_2(t): return -2*np.cos(2*np.pi*t) + 14
def lambda_3(t): return -0.5*np.cos(2*np.pi*t) + 6
def mu_1(t): return np.sin(2*np.pi*t + 5/12*np.pi) + 3
def mu_2(t): return 2*np.sin(2*np.pi*t + 5/12*np.pi) + 6
def mu_3(t): return 0.5*np.sin(2*np.pi*t + 5/12*np.pi) + 2
def omega_1(t): return np.sin(2*np.pi*t + 1/4*np.pi) + 7
def omega_2(t): return 2*np.sin(2*np.pi*t + 1/4*np.pi) + 8
def omega_3(t): return 0.5*np.sin(2*np.pi*t + 1/4*np.pi) + 4


A = [birth_death_generator(n_states[0], lambda_1, lambda t: mu_1(t) + omega_1(t)),
     birth_death_generator(n_states[1], lambda_2, lambda t: mu_2(t) + omega_2(t)),
     birth_death_generator(n_states[2], lambda_3, lambda t: mu_3(t) + omega_3(t))]


def select_slice_by_dimension(shape, dim, slice_num):
    idx = [slice(None)] * dim + [slice_num] + [slice(None)] * (len(shape) - dim - 1)
    return tuple(idx)


def control_mask(shape, i1, i2):
    mask = np.ones(shape)
    mask[select_slice_by_dimension(shape, i1, 0)] = 0
    mask[select_slice_by_dimension(shape, i2, -1)] = 0
    return mask


def control_out(i, U):  # i - MC number
    return np.sum(U[i, :][to_optimize[i, :]])


def control_in(i, U):  # i - MC number
    return np.sum(U[:, i][to_optimize[:, i]])


def f(t, U):
    f = np.zeros(n_states)
    f1 = np.full_like(f, lambda_1(t) - mu_1(t) - omega_1(t))
    f1 = f1 - U[0,1] * control_mask(n_states, 0, 1)
    f1 = f1 + U[1,0] * control_mask(n_states, 1, 0)
    f1 = f1**2

    f2 = np.full_like(f, lambda_2(t) - mu_2(t) - omega_2(t))
    f2 = f2 + U[0,1] * control_mask(n_states, 0, 1)
    f2 = f2 - U[1,0] * control_mask(n_states, 1, 0)
    f2 = f2 + U[2,1] * control_mask(n_states, 2, 1)
    f2 = f2 - U[1,2] * control_mask(n_states, 1, 2)
    f2 = f2**2

    f3 = np.full_like(f, lambda_3(t) - mu_3(t) - omega_3(t))
    f3 = f3 - U[2,1] * control_mask(n_states, 2, 1)
    f3 = f3 + U[1,2] * control_mask(n_states, 1, 2)
    f3 = f3**2

    f = f1+f2+f3

    penalty = 1e5
    f = f + penalty * U[0,1]**2 * np.abs(control_mask(n_states, 0, 1)-1.0) # penalty for outbound control at "drought" states and inbound control at "flood" states
    f = f + penalty * U[1,0]**2 * np.abs(control_mask(n_states, 1, 0)-1.0)
    f = f + penalty * U[2,1]**2 * np.abs(control_mask(n_states, 2, 1)-1.0)
    f = f + penalty * U[1,2]**2 * np.abs(control_mask(n_states, 1, 2)-1.0)

    return f


def terminal(n_states, desirable_state, factor):
    desirable_state = np.array(desirable_state)
    phiT = np.zeros(n_states)
    for idx, x in np.ndenumerate(phiT):
        diff = np.sum(np.abs(np.array(idx) - desirable_state))
        phiT[idx] = factor*diff
    return phiT


def rhs(t, phi, U):
    rhs = np.zeros_like(phi)
    for idx, x in np.ndenumerate(rhs):
        for i in range(0, len(idx)):
            select_slice = list(idx)
            select_slice[i] = slice(None)
            select_slice = tuple(select_slice)
            rhs[idx] = rhs[idx] + np.inner(A[i](t, control_in(i, U), control_out(i, U))[:, idx[i]], phi[select_slice])
    return rhs+f(t, U)


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
    ax.set_ylim(0,4)
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

def plot_value_control(state, times, values, controls,  path):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

    ax = plt.subplot(gs[0])
    ax.plot(times, values, label='value')
    ax.legend()

    ax = plt.subplot(gs[1])
    names = ['u[1,2]','u[2,1]','u[2,3]','u[3,2]']
    for k in range(0, controls.shape[1]):
        ax.plot(times, controls[:,k], label=names[k])
    ax.legend()
    ax.set_ylim(np.min(lb[to_optimize])-0.5,np.max(ub[to_optimize])+0.5)
    ax.set_title(f'({state[0]}, {state[1]}, {state[2]})', y=-0.01)

    ax_ = plt.twinx(ax)
    tsize = (np.max(times) - np.min(times))/3.0
    t = [np.min(times) + tsize*x for x in [0,1,1,2,2,3]]
    x = [state[0]+1, state[0]+1, state[1]+1, state[1]+1, state[2]+1, state[2]+1]
    ax_.fill_between(t,x, alpha=0.5)
    ax_.set_axis_off()
    ax_.set_ylim(0,4)

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
    plt.savefig(f'{path}controls_{state[0]}_{state[1]}_{state[2]}.png')
    plt.close(fig)


def print_slice(slice):
    for idx, control, val in slice:
        print(idx, v2print(control), val) #, cons1(res.x), cons2(res.x))


def pics_slice(slice, path):
    for idx, control, _ in slice:
        plot_stateandcontrol_3MCs(idx, v2m(control), path, max_level=np.max(n_states))


def pics_plots(times, values, controls, path):
    for idx, _ in np.ndenumerate(np.zeros(n_states)):
        plot_value_control(idx, times, values[tuple([slice(None)]+list(idx))], controls[tuple([slice(None)]+list(idx)+[slice(None)])], path)


def save_results(values, controls, path):
    np.save(f'{path}values.npy', values)
    np.save(f'{path}controls.npy', controls)


def load_results(path):
    values = np.load(f'{path}values.npy')
    controls = np.load(f'{path}controls.npy')
    return values, controls


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
