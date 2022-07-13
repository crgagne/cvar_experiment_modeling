import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

from task_utils import state2idcs, idcs2state


def action2str(a, num_actions=4):
    if num_actions == 2:
        if a == 0:  #  right
            return ('right')
        elif a == 1:  #  left
            return ('left')
    elif num_actions == 3:
        if a == 0:  # up
            return ('stay')
        elif a == 1:  #  right
            return ('right')
        elif a == 2:  #  left
            return ('left')
    elif num_actions == 4:
        if a == 0:  # up
            return ('up')
        elif a == 1:  #  down
            return ('down')
        elif a == 2:  #  right
            return ('right')
        elif a == 3:  #  left
            return ('left')


def plot_goal(ax, goal_idcs):
    # Plot goal location
    if goal_idcs is not None:
        if len(goal_idcs.shape) > 1:  # more than one goal?
            for goal_idx in goal_idcs:
                goal = ax.scatter(goal_idx[1], goal_idx[0], s=100, c='crimson', marker=r'$\clubsuit$', alpha=0.7)
        else:
            goal = ax.scatter(goal_idcs[1] + 0.5, goal_idcs[0] + 0.5, s=1000, c='crimson', marker=r'$\clubsuit$',
                              alpha=0.7)


def add_walls(ax, maze):
    # find wall locations #
    wall_loc_coords = np.array(np.where(maze == 1)).T  # make vertical
    for i in range(wall_loc_coords.shape[0]):
        wcoord = wall_loc_coords[i, :]
        ax.add_patch(
            patches.Rectangle((wcoord[1] - 0.5, wcoord[0] - 0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='k'))


def add_triangles(ax, maze, cm, add_labels=False, fs=5, cm_empty=10, ec=(0.4, 0.4, 0.4, 0.1), ls='-',
                  term_states_for_plotting=[], order='C'):
    # print(ls)
    triangles = {}
    labels = {}

    for x in range(maze.shape[1]):  # these are reversed
        for y in range(maze.shape[0]):

            if not idcs2state([y, x], maze, order=order) in term_states_for_plotting:

                s = 0.75  # scaling
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

                if add_labels:
                    labels[str(x) + '_' + str(y) + '_up'] = ax.annotate('', [x - 0.065, y - 0.35], fontsize=fs)
                    labels[str(x) + '_' + str(y) + '_down'] = ax.annotate('', [x - .065, y + 0.3], fontsize=fs)
                    labels[str(x) + '_' + str(y) + '_left'] = ax.annotate('', [x - 0.4, y - 0.02], fontsize=fs)
                    labels[str(x) + '_' + str(y) + '_right'] = ax.annotate('', [x + 0.25, y - 0.02], fontsize=fs)

    return (triangles, labels)


def check_for_lowest_color(color, new_q, cm, n_colors, middle_or_end='middle', roundoff=3):
    '''Middle or end is whether white is in the middle or left end of the color array'''

    if np.isnan(new_q):
        # return white (middle color)
        return (cm[int(n_colors / 2)])

    if middle_or_end == 'middle':  # white is middle color
        # check for non-zero q's; give them lowest color
        if new_q < 0 and np.round(new_q, roundoff) == 0:
            color = cm[int(n_colors / 2) - 1]
        elif new_q > 0 and np.round(new_q, roundoff) == 0:
            color = cm[int(n_colors / 2) + 1]

    return (color)


def plot_all_qvalues(Q_table, trianglelist, maze, cm, Qrange_discrete,
                     labellist=None, q_or_pi='q', roundoff=3, term_states_for_plotting=[], num_actions=4,
                     order='C',
                     min_plot=0.01):
    '''Updates a trianglelist and labellist with Q-values
    '''

    n_colors = len(cm) - 1
    statelist = np.arange(Q_table.shape[0])
    actionlist = np.arange(Q_table.shape[1])
    for s in statelist:
        for a in actionlist:

            if s not in term_states_for_plotting:

                si = state2idcs(s, maze, order=order)
                new_q = Q_table[s, a]
                color = cm[np.argmin(np.abs(new_q - Qrange_discrete))]
                if q_or_pi == 'q':
                    color = check_for_lowest_color(color, new_q, cm, n_colors, roundoff=roundoff)
                # try:
                # import pdb; pdb.set_trace()
                if np.abs(new_q) < min_plot:
                    # import pdb; pdb.set_trace()
                    trianglelist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_fc(
                        [1., 1., 1.])
                    trianglelist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_ec(None)
                    trianglelist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_visible(
                        False)
                else:
                    trianglelist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_fc(color)
                    trianglelist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_ec(None)
                # except:
                #    import pdb; pdb.set_trace()
                trianglelist[0][
                    '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_linestyle(
                    'None')
                if labellist is not None:
                    labellist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_text(
                        str(np.round(new_q, roundoff)))
                    labellist[0][
                        '_'.join(str(x) for x in si[::-1]) + '_' + action2str(a, num_actions=num_actions)].set_color(
                        'k')


def plot_q_or_pi(Q, V, title, ax, maze, q_or_pi='q', Qrange=None, roundoff=3, annot_value=False,
                 value_fontsize=8, n_colors=20,
                 inc_triangles=True, tri_ls='-', tri_ec=(0.4, 0.4, 0.4, 0.1), tri_fs=8, tri_add_labels=True,
                 plot_value=True, term_states_for_plotting=[], pi_color="blue", tri_type='all', colorbar=False,
                 order='C', min_plot=0.01):
    '''Wrapper to the plot_all_qvalues
       Can be used for pi's or Q-vales.
       Give an axis.
    '''
    # color map
    if q_or_pi == 'q':

        if Qrange is None:
            minmaxQ = np.max(np.abs(Q))
            minmaxV = np.max(np.abs(V))
            Qrange = [-1 * np.max((minmaxQ, minmaxV)), np.max((minmaxQ, minmaxV))]
            if Qrange[0] == Qrange[1]:
                Qrange = [-1, 1]

        cm_Q = sns.light_palette("red", int(n_colors / 2))[::-1] + [(0.96, 0.96, 0.96)] + sns.light_palette("green",
                                                                                                            int(n_colors / 2))
        # if invert_colors:
        #     cm_Q = cm_Q[::-1]
        #     q_range = q_range[::-1]
        Qrange_discrete = list(np.linspace(Qrange[0], -1 * Qrange[1] / (n_colors / 2), int(n_colors / 2))) + \
                          [0] + \
                          list(np.linspace(Qrange[1] / (n_colors / 2),
                                           Qrange[1], int(n_colors / 2)))
        cm_empty = int(n_colors / 2)
        # import pdb; pdb.set_trace()


    elif q_or_pi == 'pi':
        # import pdb; pdb.set_trace()
        cm_Q = [(0.96, 0.96, 0.96)] + sns.light_palette(pi_color, int(n_colors) - 1)
        cm_Q = [(1.0, 1.0, 1.0)] + sns.light_palette(pi_color, int(n_colors) - 1)
        # cm_Q = [(1, 1, 1)]+sns.color_palette("Blues",int(n_colors)-1)

        # cm_Q = [(0.96, 0.96, 0.96)]+sns.dark_palette(pi_color,int(n_colors)-1,reverse=True)

        if Qrange is None:
            maxV = np.max(np.abs(V))
            Qrange = [0, maxV]
            # print(q_range)

        Qrange_discrete = list(np.linspace(Qrange[0], Qrange[1], n_colors))
        cm_empty = 0

    assert len(cm_Q) == len(Qrange_discrete)

    # plot triangles
    trianglelist = []
    labellist = []
    if inc_triangles:
        if tri_type == 'all':
            tri, lab = add_triangles(ax, maze, cm_Q, add_labels=tri_add_labels, cm_empty=cm_empty, ec=tri_ec, ls=tri_ls,
                                     fs=tri_fs, term_states_for_plotting=term_states_for_plotting, order=order)

        trianglelist.append(tri)
        labellist.append(lab)

    if tri_add_labels == False:
        labellist = None

    # put q-values into triangles
    if inc_triangles:

        if tri_type == 'all':
            num_actions = 4
        elif tri_type == 'horiz':
            num_actions = 2

        plot_all_qvalues(Q,
                         trianglelist, maze, cm_Q, Qrange_discrete,
                         labellist=labellist, q_or_pi=q_or_pi,
                         roundoff=roundoff,
                         term_states_for_plotting=term_states_for_plotting,
                         num_actions=num_actions,
                         min_plot=min_plot)

    im_value = None
    if plot_value:

        cm_V = [(1.0, 1.0, 1.0)] + sns.light_palette('red', 10, reverse=True)
        cmap2 = matplotlib.colors.ListedColormap(cm_V)
        cMap = []
        for value, colour in zip([0, 0.001, 0.1, 0.3, .6, 1],
                                 [cm_V[0],
                                  cm_V[1], cm_V[5], cm_V[6],
                                  cm_V[7], cm_V[9]]):
            cMap.append((value, colour))
        customColourMap = LinearSegmentedColormap.from_list("custom", cMap)

        im_value = plt.imshow(V.reshape(maze.shape, order=order),
                              interpolation='none', origin='lower',
                              cmap=customColourMap,
                              vmax=Qrange[1],
                              vmin=Qrange[0],
                              alpha=0.15,
                              aspect='auto',  # this is important for not squashing
                              )

    if annot_value:
        # Loop over data dimensions and create text annotations.
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                value = np.round(V.reshape(maze.shape, order=order)[i, j], roundoff)
                if value != 0.0:
                    text = ax.text(j, i, value, fontsize=value_fontsize,
                                   ha="center", va="center", color="k")

    ax.set_title(title)
    return (trianglelist, im_value)


def embellish_plot(ax, maze, rewards, s0_py, cost, corner_labels, color_agent='b',
                   center_rewards=False, r_fontsize=8, add_rewards=True,
                   ec='k', fc='white', alpha=1, outer_lw=3, reward_color='rg', order='C'):
    add_walls(ax, maze)

    # add grids
    ax.set_yticks(np.arange(0, maze.shape[0], 1));
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True);
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, axis='both')
    ax.set_xticks(np.arange(0, maze.shape[1], 1));
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True);
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=1, axis='both')

    # add the rewards
    if add_rewards:
        reward_states = np.where(rewards != 0)[0]
        for gs in reward_states:
            # for now only one rewards per state
            rs = np.unique(rewards[gs])[0]
            # print(rs)

            if cost:
                rs = -1 * rs
            # cast to int if no dezimal places
            if rs % 1 == 0:
                r_str = str(int(rs))
            else:
                r_str = str(rs)
            # print(r_str)

            if rs > 0:
                color = sns.color_palette()[2]  # 'g'
            else:
                color = sns.color_palette()[3]  # 'r'
            if reward_color == 'k':
                color = 'k'
                if rs > 0:
                    r_str = '+' + r_str

            s_idcs = state2idcs(gs, maze, order=order)
            if center_rewards == False:
                xoffset = -0.45
            else:
                xoffset = -0.1
            ax.text(s_idcs[1] + xoffset, s_idcs[0] + 0.45, 'r = ' + r_str, fontsize=r_fontsize, color=color,
                    bbox=dict(edgecolor=ec, facecolor=fc, alpha=alpha, pad=2))

    # add in start state as dot (or agent)
    start_state = s0_py
    if start_state is not None:
        start_idcs = state2idcs(start_state, maze, order=order)
        plt.scatter(start_idcs[1] - 0.35, start_idcs[0] - 0.35, color=color_agent, marker='D', s=75)

    if corner_labels:
        # add numbers in the upper corner
        for s in range(maze.shape[0] * maze.shape[1]):
            s_idcs = state2idcs(s, maze, order=order)
            if s == start_state:
                extra = '=s0'
                color = 'k'
            else:
                extra = ''
                color = 'k'
            ax.text(s_idcs[1] - 0.45, s_idcs[0] - 0.35, str(int(s)) + extra, fontsize=7, color=color)
            # ax.text(s_idcs[0]-0.45,s_idcs[1]-0.35,str(int(s))+extra,fontsize=12,color=color)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # darken the outside
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(str(outer_lw))


def plot_sa(sa_occupancy, state_values, task, alpha_set, alpha_idx, model_name, q_or_pi='q',
            fig=None, ax=None, q_range=None, extra_title='', order='C', finish=False, round_off=2,
            pi_color='blue', min_plot=0.01, title_fs=20, start_fs=16, reward_fs=12, value_fs=14, show_alpha0=False):

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)

    if q_range is None:
        q_range = [np.min(sa_occupancy), np.max(sa_occupancy)]

    if finish:
        tri_add_labels = False
        corner_labels = False
    else:
        tri_add_labels = True
        corner_labels = True

    if np.max(state_values) > 0.01:
        plot_value = True
        annot_value = True
    else:
        plot_value = False
        annot_value = False

    print('plot_q')
    plot_q_or_pi(sa_occupancy,
                 state_values.reshape(task.maze.shape, order=order),
                 '',
                 ax,
                 task.maze,
                 q_or_pi=q_or_pi,
                 roundoff=round_off,
                 inc_triangles=True,
                 tri_add_labels=tri_add_labels,
                 annot_value=annot_value,
                 plot_value=plot_value,
                 Qrange=q_range,
                 term_states_for_plotting=task.absorbing_states,
                 n_colors=100,
                 pi_color=pi_color,
                 min_plot=min_plot,
                 value_fontsize=value_fs)
    print('embellish')
    embellish_plot(ax,
                   task.maze,
                   task.rewards,
                   None,
                   cost=False,
                   corner_labels=corner_labels,
                   color_agent='b',
                   center_rewards=False,
                   r_fontsize=reward_fs,
                   add_rewards=True,
                   ec='white',
                   fc='white',
                   alpha=0.0,
                   outer_lw=2,
                   reward_color='k')

    print('add fire labels')
    # add fire
    arr_img = mpimg.imread('../mscl/lava_pit.png')
    for fire in task.fire_locations:
        imagebox = OffsetImage(arr_img, zoom=.015)
        fire_idcs = state2idcs(fire, task.maze)
        ab = AnnotationBbox(imagebox, (fire_idcs[1], fire_idcs[0]-0.1), pad=0, frameon=False)
        ax.add_artist(ab)

    # add start label
    print('add start labels')

    start_idcs = state2idcs(task.start_location, task.maze)
    # matplotlib uses column row, so switch indcs
    ax.text(start_idcs[1], start_idcs[0], 'start', fontsize=start_fs, color='k',
            bbox=dict(edgecolor='white', facecolor='white', alpha=1, pad=2), ha='center', va='center')
    # add goal label
    print('add goal')
    for goal in task.goal_locations:
        goal_idcs = state2idcs(goal, task.maze)
        ax.text(goal_idcs[1], goal_idcs[0], 'goal', fontsize=start_fs, color='k',
                bbox=dict(edgecolor='white', facecolor='white', alpha=1, pad=2), ha='center', va='center')

    print('plot alpha')
    plt.sca(ax)
    if alpha_idx is not None:
        if show_alpha0:
            alpha_0 = r' $\alpha_0$=' + str(np.round(alpha_set[alpha_idx], 2))
        else:
            alpha_0 = r' $\alpha$=' + str(np.round(alpha_set[alpha_idx], 2))
    else:
        alpha_0 = ''
    plt.title(extra_title + model_name + alpha_0, fontsize=title_fs)
    ax.invert_yaxis()
    print('finished return fig')
    return fig
