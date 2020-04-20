
from ControlledDams import ControlledDams
from ControlledSystem import *
from functools import partial
from multiprocessing import Pool
from time import time, strftime, localtime, gmtime

np.set_printoptions(precision=3, suppress=True)

work_dir = 'D:\\projects.git\\InteractingCMCResearch\\output\\'
show_output = False

do_recalculate = False
do_probabilities = True
do_pics_by_state = True
do_pics_averages = True


n_states = [3, 4, 5]

desirable_state = [1, 1, 1]

to_optimize = np.array([[False, True, False], [True, False, True], [False, True, False]])
lb = np.array([[np.NaN, 0, np.NaN], [0, np.NaN, 0], [np.NaN, 0, np.NaN]])
ub = np.array([[np.NaN, 4, np.NaN], [4, np.NaN, 4], [np.NaN, 4, np.NaN]])

start_state = (0,3,0)

dams = ControlledDams(n_states, to_optimize, lb, ub, ['U[1->2]', 'U[2->1]', 'U[2->3]', 'U[3->2]'])









delta = 0.01
T = 1.0

if __name__ == '__main__':
    if do_recalculate:
        dams.optimize_controls(desirable_state, T, delta, show_output)
        np.save(f'{work_dir}time_mesh.npy', dams.time_mesh)
        np.save(f'{work_dir}values.npy', dams.values)
        np.save(f'{work_dir}controls.npy', dams.controls)
    else:
        dams.load_results(work_dir)

    if do_probabilities:
        dams.calculate_probabilities_theor(start_state)
        dams.calculate_probabilities_MC(start_state)
        np.save(f'{work_dir}probabilities_theor.npy', dams.probs_joint_theor)
        np.save(f'{work_dir}average_levels_theor.npy', dams.average_levels_theor)
        np.save(f'{work_dir}probabilities_MC.npy', dams.probs_joint_MC)
        np.save(f'{work_dir}average_levels_MC.npy', dams.average_levels_MC)
    else:
        dams.load_results(work_dir)

    if do_pics_by_state:
        dams.pics_plots(work_dir)
    if do_pics_averages:
        dams.pics_averages(work_dir)




