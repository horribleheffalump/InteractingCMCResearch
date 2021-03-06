
from GoulburnSimpleModel import GoulburnSimpleModel
from ControlledSystem import *
from functools import partial
from multiprocessing import Pool
from time import time, strftime, localtime, gmtime
from DELWPdata.DataApproximations import *

np.set_printoptions(precision=3, suppress=True)

work_dir = 'D:\\projects.git\\InteractingCMCResearch\\output\\'
show_output = False

do_recalculate = True
do_probabilities = True
do_pics_by_state = True
do_pics_averages = True



n_states = [5, 5, 1, 1, 1, 1]
desirable_state = [3, 3, 0, 0, 0, 0]
start_state = [3, 3, 0, 0, 0, 0]

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

    # substitute optimal control with program control
    # dams.controls = np.zeros([dams.time_mesh.shape[0]] + dams.n_states + list(dams.m2v(dams.lb).shape))
    # for idt, t in enumerate(dams.time_mesh):
    #     t_shifted = t + dams.time_shift
    #     u_SM_ref = approx_scalar(dams.SM_coeffs, t_shifted)
    #     u_Ct_ref = approx_scalar(dams.Ct_coeffs, t_shifted)
    #     u_EM_ref = approx_scalar(dams.EM_coeffs, t_shifted)
    #     u_GR_ref = approx_scalar(dams.GR_coeffs, t_shifted)
    #
    #     U_time_slice = np.zeros(dams.n_states + list(dams.m2v(dams.lb).shape))
    #     states_to_iter = np.zeros(dams.n_states)
    #     for idx, _ in np.ndenumerate(states_to_iter):
    #         U_E = u_SM_ref+u_Ct_ref+u_EM_ref+u_GR_ref
    #         U_G = [u_SM_ref, u_Ct_ref, u_EM_ref, u_GR_ref]
    #         if idx[1] == 9:
    #             U_E = 0
    #         if idx[0] == 0:
    #             U_E = 0
    #         if idx[1] == 0:
    #             U_G = [0,0,0,0]
    #         U_time_slice[idx] = np.array([U_E] + U_G)
    #     dams.controls[idt] = U_time_slice


    if do_probabilities:
        dams.calculate_probabilities_theor(start_state)
        dams.calculate_probabilities_MC(start_state)
        np.save(f'{work_dir}probabilities_theor.npy', dams.probs_joint_theor)
        np.save(f'{work_dir}average_levels_theor.npy', dams.average_levels_theor)
        np.save(f'{work_dir}probabilities_MC.npy', dams.probs_joint_MC)
        np.save(f'{work_dir}average_levels_MC.npy', dams.average_levels_MC)
    else:
        dams.load_results(work_dir)

    x_ticks = np.arange(0,1, 1.0/12) + 1.0/24
    x_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

    if do_pics_by_state:
        dams.pics_plots(work_dir, x_ticks=x_ticks, x_labels=x_labels, pic_type='png')
    if do_pics_averages:
        dams.pics_averages(work_dir, x_ticks=x_ticks, x_labels=x_labels, pic_type='png')




