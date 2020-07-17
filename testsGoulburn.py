
from GoulburnSimpleModel import GoulburnSimpleModel
from ControlledSystem import *
from functools import partial
from multiprocessing import Pool
from time import time, strftime, localtime, gmtime

np.set_printoptions(precision=3, suppress=True)

work_dir = 'D:\\projects.git\\InteractingCMCResearch\\output\\'
show_output = False

do_recalculate = False
do_probabilities = False
do_pics_by_state = True
do_pics_averages = True



n_states = [10, 10, 1, 1, 1, 1]
desirable_state = [8, 8, 0, 0, 0, 0]
start_state = [8, 8, 0, 0, 0, 0]

dams = GoulburnSimpleModel(n_states)


delta = 1e-3

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
        #np.save(f'{work_dir}probabilities_MC.npy', dams.probs_joint_theor)
        #np.save(f'{work_dir}average_levels_MC.npy', dams.average_levels_theor)
        np.save(f'{work_dir}probabilities_MC.npy', dams.probs_joint_MC)
        np.save(f'{work_dir}average_levels_MC.npy', dams.average_levels_MC)
    else:
        dams.load_results(work_dir)

    x_ticks = np.arange(0,1, 1.0/12) + 1.0/24
    x_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

    if do_pics_by_state:
        dams.pics_plots(work_dir, x_ticks=x_ticks, x_labels=x_labels)
    if do_pics_averages:
        dams.pics_averages(work_dir, x_ticks=x_ticks, x_labels=x_labels)




