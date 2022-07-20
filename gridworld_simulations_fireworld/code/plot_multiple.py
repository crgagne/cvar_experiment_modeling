import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from IPython.display import Image, display
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

from task_utils import state2idcs, idcs2state
from plotting import add_walls, plot_rewards, add_triangles


def plot_model_grid(task, alpha_plot_set, sa_occ_dict):
    """
    sa_acc_dic
        keys are modelnames, values are list of state action occupancies for each alpha
    """

    ## 1 poin = 0,0138889 inches

    n_axes = len(sa_occ_dict)
    # 6X6 inch axes
    fig, axes = plt.subplots(1, n_axes, figsize=(6*n_axes, 6), dpi=100)
    # TODO: fix for only one axes
    for i, (model_name, sa_occ) in enumerate(sa_occ_dict.items()):
        axes[i] = plot_multiple_alphas(ax=axes[i], task=task, sa_occ_list=sa_occ, model_name=model_name, alpha_plot_set=alpha_plot_set)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    matplotlib.pyplot.tick_params(left=False, bottom=False)
    return fig, axes


def plot_multiple_alphas(ax, task, sa_occ_list, model_name, alpha_plot_set):
    # set up the gridworl
    plot_gridworld(ax, task)
    # plot the trajectories
    plot_trajectories(ax=ax, task=task, sa_occ_list=sa_occ_list, alpha_plot_set=alpha_plot_set)

    return ax


def plot_gridworld(ax, task, add_rewards=True):
    """
    Parameters
    ----------
    ax : matplotlib axes
        ax to be adjusted
    task: task_fire_world object
    rewards :
        list of rewards for every state
    add_rewards : boolean
        indicate whether to plot reward values

    Returns
    -------

    """
    maze = task.maze

    box_size = 6/len(maze[0])
    size_in_pixels = 600
    pixels_per_box = 600/len(maze[0])


    reward_fontsize = int((box_size/6)*72)
    state_fontsize = int((box_size/5)*72)

    # add walls
    add_walls(ax, maze)

    # add grids
    ax.set_yticks(np.arange(0, maze.shape[0], 1))
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, axis='both')
    ax.set_xticks(np.arange(0, maze.shape[1], 1))
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, axis='both')

    # add the rewards
    print('re')
    if add_rewards:
        plot_rewards(ax=ax, task=task, reward_fontsize=reward_fontsize)

    print('start')
    # plot start state
    start_idcs = state2idcs(task.start_location, maze, order=task.order)
    ax.text(start_idcs[1], start_idcs[0], 'start', fontsize=state_fontsize, color='k',
            bbox=dict(edgecolor='white', facecolor='white', alpha=1, pad=2), ha='center', va='center')

    # plot goal locations
    for goal in task.goal_locations:
        goal_idcs = state2idcs(goal, task.maze)
        ax.text(goal_idcs[1], goal_idcs[0], 'goal', fontsize=state_fontsize, color='k',
                bbox=dict(edgecolor='white', facecolor='white', alpha=1, pad=2), ha='center', va='center')

    # add fire labels
    # width 3958 pixels
    # height 1991
    zoom = pixels_per_box/3958/1.5
    arr_img = mpimg.imread('../mscl/lava_pit.png')
    for fire in task.fire_locations:
        imagebox = OffsetImage(arr_img, zoom=zoom)
        fire_idcs = state2idcs(fire, task.maze)
        ab = AnnotationBbox(imagebox, (fire_idcs[1], fire_idcs[0] - 0.1), pad=0, frameon=False)
        ax.add_artist(ab)

    # darken the outside
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(str(2))
    # flip everything
    ax.invert_yaxis()


def plot_trajectories(ax, task, sa_occ_list, alpha_plot_set):

    print(sa_occ_list)
    # plot triangles

    n_colors = 20
    cm_Q = [(1.0, 1.0, 1.0)] + sns.light_palette('blue', int(n_colors) - 1)
    print(cm_Q)
    # cm_Q = [(1, 1, 1)]+sns.color_palette("Blues",int(n_colors)-1)

    # cm_Q = [(0.96, 0.96, 0.96)]+sns.dark_palette(pi_color,int(n_colors)-1,reverse=True)


    maxV = np.max(np.abs(V))
    Qrange = [0, maxV]
    print(Qrange)

    Qrange_discrete = list(np.linspace(Qrange[0], Qrange[1], n_colors))
    cm_empty = 0


    maze = task.maze
    triangles = {}
    labels = {}

    for x in range(maze.shape[1]):  # these are reversed
        for y in range(maze.shape[0]):

            # only plot non terminal states
            if not idcs2state([y, x], maze, order=task.order) in task.absorbing_states:

                # number of triangles

                # scale
                s = 0.75
                triangles[str(x) + '_' + str(y) + '_up'] = plt.Polygon(
                    [[x - 0.2 * s, y - 0.25], [x + 0.2 * s, y - 0.25], [x + 0, y - 0.45]], fc=cm[cm_empty], ec=ec,
                    alpha=1, linestyle=ls)
                triangles[str(x) + '_' + str(y) + '_down'] = plt.Polygon(
                    [[x - 0.2 * s, y + 0.25], [x + 0.2 * s, y + 0.25], [x + 0, y + 0.45]], fc=cm[cm_empty], ec=ec,
                    alpha=1, linestyle=ls)
                triangles[str(x) + '_' + str(y) + '_left'] = plt.Polygon(
                    [[x - 0.25, y - 0.2 * s], [x - 0.25, y + 0.2 * s], [x - 0.45, y + 0]], fc=cm[cm_empty], ec=ec,
                    alpha=1, linestyle=ls)
                triangles[str(x) + '_' + str(y) + '_right'] = plt.Polygon(
                    [[x + 0.25, y - 0.2 * s], [x + 0.25, y + 0.2 * s], [x + 0.45, y + 0]], fc=cm[cm_empty], ec=ec,
                    alpha=1, linestyle=ls)

                ax.add_patch(triangles[str(x) + '_' + str(y) + '_up'])
                ax.add_patch(triangles[str(x) + '_' + str(y) + '_down'])
                ax.add_patch(triangles[str(x) + '_' + str(y) + '_left'])
                ax.add_patch(triangles[str(x) + '_' + str(y) + '_right'])



    return (triangles, labels)



