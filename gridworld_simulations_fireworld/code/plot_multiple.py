import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from IPython.display import Image, display
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

from task_utils import state2idcs, idcs2state, action_str_to_num
from plotting import add_walls, plot_rewards, add_triangles


def plot_model_grid(task, alpha_plot_set, sa_occ_dict):
    """
    sa_acc_dic
        keys are modelnames, values are list of state action occupancies for each alpha
    """

    ## 1 poin = 0,0138889 inches

    n_axes = len(sa_occ_dict)
    # 6X6 inch axes
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 6), dpi=100)
    # TODO: fix for only one axes
    for i, (model_name, sa_occ) in enumerate(sa_occ_dict.items()):
        axes[i], legend_elements, legend_labels = plot_multiple_alphas(ax=axes[i], task=task, sa_occ_list=sa_occ, model_name=model_name,
                                       alpha_plot_set=alpha_plot_set)


    plt.tight_layout()
    #fig.legend(legend_elements, legend_labels)

    return fig, axes


def plot_multiple_alphas(ax, task, sa_occ_list, model_name, alpha_plot_set):
    # set up the gridworl
    plot_gridworld(ax, task)
    # plot the trajectories
    ax.set_title(model_name)
    legend_labels, legend_elements = plot_trajectories(ax=ax, task=task, sa_occ_list=sa_occ_list, alpha_plot_set=alpha_plot_set)
    return ax, legend_elements, legend_labels


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

    box_size = 6 / len(maze[0])
    size_in_pixels = 600
    pixels_per_box = 600 / len(maze[0])

    reward_fontsize = int((box_size / 6) * 72)
    state_fontsize = int((box_size / 5) * 72)

    # add walls
    add_walls(ax, maze)

    # add grids
    ax.set_yticks(np.arange(0, maze.shape[0], 1))
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, axis='both')
    ax.set_xticks(np.arange(0, maze.shape[1], 1))
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, axis='both')

    # remove ticks and keep grid

    ax.tick_params(
        axis='both',  #
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the left edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False)
    # add the rewards
    if add_rewards:
        plot_rewards(ax=ax, task=task, reward_fontsize=reward_fontsize)

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
    zoom = pixels_per_box / 3958 / 1.5
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
    # plot triangles
    n_colors = 5
    possible_colors = ['blue', 'purple', 'green', 'red']
    color_maps = []
    legend_elements = []
    legend_labels = []
    # create color palettes and legend for each alpha
    for i in range(len(alpha_plot_set)):
        color_maps.append([(1.0, 1.0, 1.0)] + sns.light_palette(possible_colors[i], int(n_colors)))
        legend_elements.append(patches.Patch(
                    fc=color_maps[-1][-1], ec=(0.4, 0.4, 0.4, 0.1), alpha=1, linestyle='-'))
        legend_labels.append(f'alpha = {np.around(alpha_plot_set[i], 2)}')

    ax.legend(legend_elements, legend_labels)

    # sort sa_occs into discrete bins to determine color
    # make the lowest bin negative so it will not be used,
    # because lowest bin of the color palette is gray
    start_bin = -1/n_colors
    color_bins = np.linspace(start_bin, 1, n_colors+2)
    sa_occ_idx = np.digitize(sa_occ_list, color_bins, right=True) - 1
    maze = task.maze

    for x in range(maze.shape[1]):  # these are reversed
        for y in range(maze.shape[0]):

            # only plot non terminal states
            state = idcs2state([y, x], maze, order=task.order)
            if state not in task.absorbing_states:
                # print(len(sa_occ_list))
                # print(sa_occ_list[0])
                # get all triangles to be plotted
                triangles_per_action = []
                for action in range(4):
                    triangles_per_alpha = []
                    for i in range(len(alpha_plot_set)):
                        # get color for each triangle
                        # print(len(color_maps[0]))
                        # print(sa_occ_idx[i][state, action])
                        color_bin = sa_occ_idx[i][state, action]
                        triangles_per_alpha.append(color_bin)
                    triangles_per_action.append(triangles_per_alpha)

                # total width for all triangles to be plotted next to each other
                total_width = 0.7
                # UP
                triangles_up = triangles_per_action[action_str_to_num('up')]
                # check how many traingles need to be plotted and adjust width accordingly
                n_up = np.count_nonzero(np.asarray(triangles_up)-1)
                if n_up > 0:
                    single_width = total_width / n_up
                else:
                    single_width = total_width
                # iterate over triangles for each alpha
                n = 0
                for i, color_bin in enumerate(triangles_up):
                    color = color_maps[i][color_bin]
                    if color_bin > 1:
                        # go half total width to the side and add up i*single width for left offset
                        ax.add_patch(plt.Polygon([[x - total_width/2 + n*single_width, y - 0.25],
                                                  [x - total_width/2 + (n + 1) * single_width, y - 0.25],
                                                  [x - total_width/2 + n*single_width + single_width/2, y - 0.45]],
                                                 fc=color, ec=(0.4, 0.4, 0.4, 0.1), alpha=1, linestyle='-'))
                        n += 1


                # DOWN
                triangles_down = triangles_per_action[action_str_to_num('down')]
                n_down = np.count_nonzero(np.asarray(triangles_down)-1)
                if n_down > 0:
                    single_width = total_width / n_down
                else:
                    single_width = total_width
                # iterate over triangles for each alpha
                # count actually plotted trinagles separately, i is needed for the corret color map
                n = 0
                for i, color_bin in enumerate(triangles_down):
                    color = color_maps[i][color_bin]
                    if color_bin > 1:
                        # go half total width to the side and add up i*single width for left offset
                        ax.add_patch(plt.Polygon([[x - total_width/2 + n*single_width, y + 0.25],
                                                  [x - total_width/2 + (n + 1) * single_width, y + 0.25],
                                                  [x - total_width/2 + n*single_width + single_width/2, y + 0.45]],
                                                 fc=color, ec=(0.4, 0.4, 0.4, 0.1), alpha=1, linestyle='-'))
                        n += 1

                # RIGHT
                # for left and right switch width and heigt compared to up an down
                triangles_right = triangles_per_action[action_str_to_num('right')]
                n_right = np.count_nonzero(np.asarray(triangles_right)-1)
                if n_right > 0:
                    single_width = total_width / n_right
                else:
                    single_width = total_width
                # iterate over triangles for each alpha
                n = 0
                for i, color_bin in enumerate(triangles_right):
                    color = color_maps[i][color_bin]
                    if color_bin > 1:
                        # go half total width to the side and add up i*single width for left offset
                        ax.add_patch(plt.Polygon([[x + 0.25, y - total_width/2 + n*single_width],
                                                  [x + 0.25, y - total_width/2 + (n + 1) * single_width],
                                                  [x + 0.45, y - total_width/2 + n*single_width + single_width/2]],
                                                 fc=color, ec=(0.4, 0.4, 0.4, 0.1), alpha=1, linestyle='-'))
                        n += 1


                # LEFT
                # for left and right switch width and heigt compared to up an down
                triangles_left = triangles_per_action[action_str_to_num('left')]
                # no color is index one, so substract 1
                n_left = np.count_nonzero(np.asarray(triangles_left)-1)
                if n_left > 0:
                    single_width = total_width / n_left
                else:
                    single_width = total_width
                # iterate over triangles for each alpha
                n = 0
                for i, color_bin in enumerate(triangles_left):
                    color = color_maps[i][color_bin]
                    if color_bin > 1:
                        # go half total width to the side and add up i*single width for left offset
                        ax.add_patch(plt.Polygon([[x - 0.25, y - total_width/2 + n*single_width],
                                     [x - 0.25, y - total_width/2 + (n + 1) * single_width],
                                     [x - 0.45, y - total_width/2 + n*single_width + single_width/2]],
                                    fc=color, ec=(0.4, 0.4, 0.4, 0.1), alpha=1, linestyle='-'))
                        n += 1
    return legend_labels, legend_elements



