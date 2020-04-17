from functools import partial

from BasicCMC import cronsum
from ControlledSystem import *
from scipy.optimize import minimize
from multiprocessing import Pool
from time import time, strftime, localtime, gmtime

work_dir = 'D:\\projects.git\\InteractingCMCResearch\\output\\'
show_output = False

do_recalculate = False
do_probabilities = False
do_pics_by_state = False
do_pics_averages = True

start_state = (0,3,0)

np.set_printoptions(precision=3, suppress=True)

init = m2v(lb)
bounds = tuple([(m2v(lb)[i], m2v(ub)[i]) for i in range(0, len(m2v(lb)))])

tol_constraints = 1e-10
consts = []
consts.append({'type': 'ineq', 'fun': lambda u: -np.abs(v2m(u)[0, 1] * v2m(u)[1, 0]) + tol_constraints})
consts.append({'type': 'ineq', 'fun': lambda u: -np.abs(v2m(u)[1, 2] * v2m(u)[2, 1]) + tol_constraints})



def proc(idx, t, phi):
    res = minimize(lambda x: rhs_tensor(t, phi, v2m(x))[idx], init, bounds=bounds, constraints=consts) #, options={'disp':True}, method='SLSQP')
    return idx, res.x, res.fun



delta = 0.01
T = 1.0
time_mesh = np.arange(T, 0.0 - delta / 2, -delta)

if __name__ == '__main__':
    if do_recalculate:
        pool = Pool(processes=8)
        phi = terminal(n_states, desirable_state, 2)
        phi_enumerate = [x[0] for x in list(np.ndenumerate(phi))]
        values = np.zeros([time_mesh.shape[0]] + n_states)
        controls = np.zeros([time_mesh.shape[0]] + n_states + list(m2v(lb).shape))

        time_start = time()
        for idt, t in enumerate(time_mesh):
            slice = pool.map(partial(proc, t=t, phi=phi), phi_enumerate)
            phi = phi + delta * slice2dphi(slice)
            time_elapsed = time() - time_start
            time_step_average = time_elapsed / (idt + 1)
            time_rest = time_step_average * len(time_mesh) - time_elapsed
            time_finish = time() + time_rest
            print(f'step: {t}, elapsed: {strftime("%H:%M:%S", gmtime(time_elapsed))}, average step: {time_step_average:.3f} sec, approximate finish time: {strftime("%d %b %Y %H:%M:%S", localtime(time_finish))} ({strftime("%H:%M:%S", gmtime(time_rest))} rest)')
            if show_output:
                print_slice(slice)
            values[len(time_mesh)-idt-1,] = phi
            controls[len(time_mesh)-idt-1,] = slice2U(slice)

        save_results(values, controls, work_dir)
    if do_probabilities:
        _, controls, _, _ = load_results(work_dir)
        N = 1000
        pool = Pool(processes=8)
        paths = pool.map(partial(sample_path, times=np.flip(time_mesh), controls=controls, x0=start_state), range(0, N))
        paths = np.array(paths)

        probs_joint = np.zeros([time_mesh.shape[0]] + n_states)
        # average_levels = np.zeros([time_mesh.shape[0]] + [len(n_states)])
        average_levels = np.average(np.array(paths), axis=0)

        for idt, _ in enumerate(time_mesh):
            p = np.zeros(n_states)
            for idp, _ in np.ndenumerate(p):
                p[idp] = 1.0 - np.count_nonzero(np.linalg.norm(paths[:,idt,:]-np.array(idp), axis=1))/N
            probs_joint[idt] = p
            # average_levels[idt,] = average_level(p)
        np.save(f'{work_dir}probabilities.npy', probs_joint)
        np.save(f'{work_dir}average_levels.npy', average_levels)

        # incorrect Kolmogorov equation solution
        # #controls = np.zeros([time_mesh.shape[0]] + n_states + list(m2v(lb).shape))
        # probs_joint = np.zeros([time_mesh.shape[0]] + n_states)
        # average_levels = np.zeros([time_mesh.shape[0]] + [len(n_states)])
        # p = np.zeros(n_states)
        # p[start_state] = 1.0
        # print(0, np.sum(p), np.min(p), np.max(p))
        # probs_joint[0,] = p
        # average_levels[0,] = average_level(p)
        # for idt, t in enumerate(np.flip(time_mesh)):
        #     if t>0:
        #         dp = np.zeros_like(p)
        #         for idp, _ in np.ndenumerate(p):
        #             control_v = controls[idt,][idp]
        #             dp[idp] = rhs_p_tensor(t, p, v2m(control_v))[idp] ## THIS IS NOT CORRECT!!! NEED TO CALCULATE THE TRANSITION TENSOR!!!!
        #         p = p + delta * dp
        #         print(t, np.sum(p), np.min(p), np.max(p))
        #         min_p = np.min(p)
        #         if min_p < 0:
        #             p = p - p * (p<0)
        #         p = p / np.sum(p)
        #         probs_joint[idt,] = p
        #         average_levels[idt,] = average_level(p)
        # np.save(f'{work_dir}probabilities.npy', probs_joint)
    if do_pics_by_state:
        values, controls, probabilities, _ = load_results(work_dir)
        pics_plots(np.flip(np.arange(T, 0.0 - delta / 2, -delta)), values, probabilities, controls, work_dir)
    if do_pics_averages:
        _, _, _, average_levels = load_results(work_dir)
        pics_averages(np.flip(np.arange(T, 0.0 - delta / 2, -delta)), average_levels, work_dir)

    #pics_plots(values, controls, work_dir)

    #print_controls(controls)
    #pics_slice(results[0.9], work_dir)
    #print()




