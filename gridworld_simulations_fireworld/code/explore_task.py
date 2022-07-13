import numpy as np
import pickle
import sys
import os
import time

import matplotlib.pyplot as plt

import dp
from dp import CVaR_DP
import mcmc_eval
from mcmc_eval import run_simulation, calc_V_CVaR_MCMC
import plotting
from plotting import plot_sa
import example_tasks
import gc


# fireworld_1: first implementation of the environment, allows for doing nothing by moving outside the world
# fireworld_2: different branch, does not allow moving outside the world

save_stem = 'fireworld_1'


# Wrapper functions for computing policies, generating behavior and plotting for a selected task


def main():
    # select task to use
    task = example_tasks.task_mini
    only_plot = False
    gamma = 0.9
    time_horizon = 5
    parallel = False

    interpolation_set = np.array([0., 0.01, 0.01274275, 0.01623777, 0.02069138,
                                  0.02636651, 0.03359818, 0.04281332, 0.05455595, 0.06951928,
                                  0.08858668, 0.11288379, 0.14384499, 0.18329807, 0.23357215,
                                  0.29763514, 0.37926902, 0.48329302, 0.61584821, 0.78475997,
                                  1.])

    # choose a subset of alphas
    alpha0_i_set = [1, 11, 14, 16, 19]
    alpha0_set = [interpolation_set[i] for i in alpha0_i_set]
    # alpha_set = [interpolation_set[i] for i in alpha0_i_set]
    alpha_set = interpolation_set
    alpha_plot_set = alpha0_set
    # free up memory

    if not os.path.isdir('../saved_figures/' + save_stem):
        os.mkdir('../saved_figures/' + save_stem)

    if not os.path.isdir('../saved_results/' + save_stem):
        os.mkdir('../saved_results/' + save_stem)

    # compute policies
    if not only_plot:
        path_dict = compute_policies(task=task, task_name=task.task_name, alpha_set=alpha_set, alpha0_set=alpha0_set,
                                     time_horizon=time_horizon, model_names=['fCVaR'], gamma=gamma, parallel=parallel)

    # create behavior and plot
    plot_task(task=task, task_name=task.task_name, alpha_set=alpha_set, alpha0_set=alpha0_set,
              alpha_plot_set=alpha_plot_set, model_names=['fCVaR'])
    return None


def compute_policies(task,
                     task_name,
                     alpha_set,
                     alpha0_set,  # needed for fCVaR
                     time_horizon=60,
                     gamma=0.9,
                     model_names=['pCVaR', 'nCVaR', 'fCVaR'],
                     verbose=True,
                     parallel=False,
                     ):
    save_stem = 'fireworld_1'
    # create folders for results if necessary
    if not os.path.isdir('../saved_figures/' + save_stem + '/' + task_name):
        os.mkdir('../saved_figures/' + save_stem + '/' + task_name)

    if not os.path.isdir('../saved_results/' + save_stem + '/' + task_name):
        os.mkdir('../saved_results/' + save_stem + '/' + task_name)

    # keep track of save locations
    save_dict = {}
    # run for different models
    for model_name in model_names:
        print('Computing policies for {}'.format(model_name))
        start_time = time.time()
        # fCVaR needs extra treatment
        if model_name == 'fCVaR':
            qs = []
            for alpha0 in alpha0_set:
                start_time_alpha = time.time()
                print(time_horizon)
                results = CVaR_DP(task, T=time_horizon, alpha0=alpha0, alpha_set=alpha_set, cvar_type=model_name,
                                  gamma=0.9, verbose=True, parallel=parallel)
                qs.append(results['Q_CVaR'])
                time_taken_alpha = time.time() - start_time_alpha
                print('Policies Computed for {} with alpha={} in {} seconds'.format(model_name, alpha0, time_taken_alpha))

            results = {}
            results['Q_CVaR'] = np.moveaxis(np.array(qs), 0, -1)

        else:
            results = CVaR_DP(task, T=time_horizon, alpha_set=alpha_set, cvar_type=model_name,
                              gamma=gamma, verbose=verbose, parallel=parallel)
            print('policy shape', results['pi'].shape)

        savename = '../saved_results/' + save_stem + '/' + task_name + '/' + model_name + '_T=' + str(time_horizon) + \
                   '_interpset=' + str(len(alpha_set)) + '.p'
        # saving results
        pickle.dump(results, open(savename, "wb"))
        # free up results
        del results
        gc.collect()
        save_dict[model_name] = savename

        time_taken = time.time() - start_time
        print('Policies Computed for {} in {} seconds'.format(model_name, time_taken))

    print('Finished successfully')
    return save_dict


def generate_behavior(task, savename, alpha_set, alpha0_set, time_horizon=60, model_name='pCVaR', alpha0=1.0,
                      invtemp=10, Nsims=1000, gamma=0.9):
    # get the appropriate alpha
    alpha0_i = np.where(np.asarray(alpha_set) == alpha0)[0][0]
    # print(np.where(np.asarray(alpha0_set)==alpha0))
    # load Q-function
    results = pickle.load(open(savename, "rb"))

    # get the q-values
    Q = results['Q_CVaR']
    if model_name == 'fCVaR':
        Q = Q[:, :, :, :, np.where(np.asarray(alpha0_set) == alpha0)[0][0]]

    # calculate policy using inverse temperature
    if invtemp == 'max':
        policy = np.zeros_like(Q)
        for s in range(policy.shape[0]):
            for alph in range(policy.shape[2]):
                for t in range(policy.shape[3]):
                    maxQ_i = np.argmax(Q[s, :, alph, t])
                    policy[s, maxQ_i, alph, t] = 1.0
    else:
        policy = np.zeros_like(Q)
        denom = np.sum(np.exp(invtemp * Q[:, :, :, :]), axis=1)
        for a in range(4):
            policy[:, a, :, :] = np.exp(invtemp * Q[:, a, :, :]) / denom

    # generate behavior
    if model_name == 'pCVaR':
        Xis = results['Xis']
        adjust_alpha = True
    else:
        Xis = None
        adjust_alpha = False

    # run the MCMC for the changing policy

    results_mcmc = run_simulation(task,
                                  policy,  # must be states x actions x alphas x time-steps
                                  gamma,
                                  T=time_horizon,
                                  n_sims=Nsims,
                                  s0=task.start_location,
                                  alpha_i0=alpha0_i,
                                  alpha_set=alpha_set,
                                  Xis=Xis,
                                  adjust_alpha=adjust_alpha)

    # hopefully free up some memory

    del Q
    return results_mcmc, results


def plot_task(task,
              task_name,
              alpha_set,
              alpha0_set,
              alpha_plot_set,
              time_horizon=60,
              model_names=['pCVaR', 'nCVaR', 'fCVaR'], ):
    # set seed
    np.random.seed(1)
    save_stem_task = task_name

    # iterate over models
    for model_name in model_names:

        for alpha in alpha_plot_set:

            n_sims = 1000
            invtemp = 'max'
            alpha0_i = np.where(np.asarray(alpha_set) == alpha)[0][0]

            savename = '../saved_results/' + save_stem + '/' + save_stem_task + '/' + model_name + '_T=' + str(
                time_horizon) + \
                       '_interpset=' + str(len(alpha_set)) + '.p'
            # print(savename)

            print('Generating behavior for {} with alpha={}'.format(model_name, alpha))
            # print(task)
            results_mcmc, results = generate_behavior(task=task,
                                                      savename=savename,
                                                      alpha_set=alpha_set,
                                                      alpha0_set=alpha0_set,
                                                      model_name=model_name,
                                                      alpha0=alpha,
                                                      invtemp=invtemp,
                                                      Nsims=n_sims)

            # print('Finished Generating behavior')
            print('Plotting trajectories...')
            # sum over all simulations, drop terminal state, get average by dividing through nsims
            sa_occupancy = results_mcmc['first_sa_occup'].sum(axis=0)[0:-1, :] / n_sims

            if model_name == 'pCVaR':
                # special stuff for pcvar
                med_alpha_per_state = []
                mean_alpha_per_state = []
                for s in range(task.n_states):
                    med_alpha_per_state.append(np.median(results_mcmc['alphas'][results_mcmc['states'] == s]))
                    mean_alpha_per_state.append(np.mean(results_mcmc['alphas'][results_mcmc['states'] == s]))
                med_alpha_per_state = np.array(med_alpha_per_state)
                mean_alpha_per_state = np.array(mean_alpha_per_state)
                max_sa_occupancy = np.max(sa_occupancy, axis=1)  # [0:(task.Ns-1)]#.reshape(task.maze.shape)
                med_alpha_per_state_masked = med_alpha_per_state[0:(task.n_states - 1)].copy()
                med_alpha_per_state_masked[max_sa_occupancy <= 0.05] = 0.0

                fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
                fig = plot_sa(sa_occupancy,
                              med_alpha_per_state_masked,  # np.round(med_alpha_per_state[0:(task.Ns-1)],2),
                              task,
                              alpha_set,
                              alpha0_i,
                              model_name=model_name,
                              q_or_pi='pi',
                              fig=fig,
                              ax=ax,
                              finish=True,
                              pi_color='blue',
                              min_plot=0.05, title_fs=26, start_fs=18, reward_fs=12, value_fs=14, show_alpha0=True)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
                fig = plot_sa(sa_occupancy,
                              np.zeros(task.n_states - 1),
                              task,
                              alpha_set,
                              alpha0_i,
                              model_name=model_name,
                              q_or_pi='pi',
                              fig=fig,
                              ax=ax,
                              finish=True,
                              pi_color='blue', min_plot=0.05, title_fs=26, start_fs=18, reward_fs=12, show_alpha0=False)

            # save plot
            # make directory for each model
            save_dir = f'../saved_figures/{save_stem}/{task_name}/{model_name}'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            # store plot for each alpha
            fig_savepath = f'{save_dir}/alpha={alpha}.png'
            fig.savefig(fig_savepath, format='png')
            print(f'saved figure for {model_name} with alpha: {alpha} in {fig_savepath}')

            del fig
            del results
            del results_mcmc
            gc.collect()


if __name__ == '__main__':
    main()
