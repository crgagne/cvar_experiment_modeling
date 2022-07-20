from plot_multiple import plot_multiple_alphas, plot_gridworld, plot_model_grid
from explore_task import plot_task_in_one
import example_tasks
import numpy as np
import matplotlib.pyplot as plt


def main():
    task = example_tasks.task_mini
    interpolation_set = np.array([0., 0.01, 0.01274275, 0.01623777, 0.02069138,
                                  0.02636651, 0.03359818, 0.04281332, 0.05455595, 0.06951928,
                                  0.08858668, 0.11288379, 0.14384499, 0.18329807, 0.23357215,
                                  0.29763514, 0.37926902, 0.48329302, 0.61584821, 0.78475997,
                                  1.])
    alpha_set = interpolation_set
    alpha_plot_index = [4, 11, 16, 19]
    alpha_plot_set = interpolation_set[alpha_plot_index]
    alpha0_i_set = [1, 11, 14, 16, 19]
    alpha0_set = [interpolation_set[i] for i in alpha0_i_set]

    plot_task_in_one(task=task, alpha0_set=alpha0_set, alpha_set=alpha_set,
                     alpha_plot_set=alpha_plot_set, model_names=['pCVaR', 'nCVaR'], time_horizon=5)

    plt.show()

if __name__ == '__main__':
    main()
