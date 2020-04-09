from functools import partial
from ControlledSystem import *
from scipy.optimize import minimize
from multiprocessing import Pool

work_dir = 'D:\\pycharm.git\\InteractingCMCResearch\\output\\'
recalculate = False

np.set_printoptions(precision=3, suppress=True)

init = m2v(lb)
bounds = tuple([(m2v(lb)[i], m2v(ub)[i]) for i in range(0, len(m2v(lb)))])

tol_constraints = 1e-10
consts = []
consts.append({'type': 'ineq', 'fun': lambda u: -np.abs(v2m(u)[0, 1] * v2m(u)[1, 0]) + tol_constraints})
consts.append({'type': 'ineq', 'fun': lambda u: -np.abs(v2m(u)[1, 2] * v2m(u)[2, 1]) + tol_constraints})



def proc(idx, t, phi):
    res = minimize(lambda x: rhs(t, phi, v2m(x))[idx], init, bounds=bounds, constraints=consts) #, options={'disp':True}, method='SLSQP')
    return idx, res.x, res.fun

delta = 0.01
T = 1.0

if recalculate:
    if __name__ == '__main__':
        pool = Pool(processes=8)
        phi = terminal(n_states, desirable_state, 2)
        phi_enumerate = [x[0] for x in list(np.ndenumerate(phi))]
        time = np.arange(T, 0.0-delta/2, -delta)
        #time = [1.0]
        values = np.zeros([time.shape[0]] + n_states)
        controls = np.zeros([time.shape[0]] + n_states + list(m2v(lb).shape))


        for idt, t in enumerate(time):
            slice = pool.map(partial(proc, t=t, phi=phi), phi_enumerate)
            phi = phi + delta * slice2dphi(slice)
            print(t)
            print_slice(slice)
            values[len(time)-idt-1,] = phi
            controls[len(time)-idt-1,] = slice2U(slice)

        save_results(values, controls, work_dir)
else:
    values, controls = load_results(work_dir)
    pics_plots(np.flip(np.arange(T, 0.0 - delta / 2, -delta)), values, controls, work_dir)

    #pics_plots(values, controls, work_dir)

    #print_controls(controls)
    #pics_slice(results[0.9], work_dir)
    #print()




