import numpy as np
from scipy.stats import norm
from task_utils import state2idcs, build_P_from_maze, return_same_state_if_boundary, state2idcs, action_str_to_num


class TaskFireWorld:

    # Example Diagram 5x5#
    #  [_, _, _, _, _]
    #  [_, _, _, _, _]
    #  [S, _, L, _, G]
    #  [_, _, _, _, _]
    #  [_, _, _, _, _]

    def __init__(self, nrows=5,
                 ncols=5,
                 start_location=10,
                 fire_locations=[12],
                 goal_locations=[14],
                 wall_locations=[],
                 task_name=None,
                 err_prob=0.05,
                 step_penalty=-0.1,
                 noise_mode='adjacent_noisy',
                 fire_reward=-10,
                 order='C'):
        """

        Parameters
        ----------
        nrows : int
            number of rows
        ncols: int
            number of columns
        start_location: int
            starting location of agent
        fire_locations: list
            location of firepits
        goal_locations : list
            location of goal
        wall_locations : list
            location of walls
        err_prob : float
            probability of unintended action
        noise_mode : str
            either 'adjacent noisy' or ...
        fire_reward
            value of fire
        order
            either 'C': columns first or rows first
        """
        self.task_name = task_name
        # number of states
        self.n_states = nrows * ncols + 1  # with terminal state
        # array representing world
        self.maze = np.zeros((nrows, ncols))
        # number of actions
        self.n_actions = 4
        # probability of unintended action
        self.err_prob = err_prob
        self.step_penalty = step_penalty
        # noise mode
        self.noise_mode = noise_mode
        self.start_location = start_location
        self.fire_locations = fire_locations
        self.goal_locations = goal_locations
        self.wall_locations = wall_locations
        # terminal state is not on the grid, all absorbing/reward states lead to terminal state
        self.terminal_state = self.n_states - 1
        # all states that lead to terminal state, ie. goal and fire
        self.absorbing_states = self.fire_locations + self.goal_locations
        # rewards for absorbing states
        # rewards are proportional to number of steps, start with 0
        self.goal_reward = 5
        self.fire_reward = fire_reward
        self.r_min = fire_reward
        # may cause problems later
        self.r_max = self.goal_reward
        # ?? reward step size
        self.dr = 1
        self.reward_dicts = {}
        for loc in self.goal_locations:
            self.reward_dicts[loc] = self.goal_reward
        for loc in self.fire_locations:
            self.reward_dicts[loc] = self.fire_reward

        # add maze with state labels
        self.maze_w_state_labels = self.maze.copy()
        for state in range(self.n_states - 1):
            idcs = state2idcs(state, self.maze, order=order)
            self.maze_w_state_labels[idcs[0], idcs[1]] = state

        # add wall states to maze
        for s in self.wall_locations:
            idcs = state2idcs(s, self.maze, order=order)
            self.maze[idcs[0], idcs[1]] = 1

        # state transition matrix ##
        self.P = build_P_from_maze(self.maze,
                                   self.n_states,
                                   self.n_actions,
                                   self.noise_mode,
                                   self.err_prob,
                                   self.absorbing_states,
                                   self.terminal_state,
                                   order=order)

        # reward range and possible values
        self.r_support = np.append(np.arange(self.r_min, self.r_max + self.dr, self.dr), self.step_penalty) # [-1,0,1,2,3]
        #print(self.r_support)

        self.n_rewards = len(self.r_support)

        # create reward distributions per state (or deterministc in this case),
        # and a list of rewards associated with each state
        self.p_rewards = np.zeros((self.n_states, self.n_rewards))
        self.rewards = np.zeros(self.n_states)

        for state in range(self.n_states):
            if state in self.reward_dicts.keys():
                r = self.reward_dicts[state]

            else:
                r = self.step_penalty  # fill in all non-rewarded states with reward = -0.1, find suitable value
            self.rewards[state] = r
            # print(self.rewards)
            self.p_rewards[state, np.where(self.r_support == r)] = 1.0  # deterministic rewards
            #print(np.where(self.r_support==r))

    # def states_allowed_at_time(self, t):
    #    return ([s for s in range(self.n_states)])

    def __str__(self):
        # IMPORTANT: copy rewads to avoid side effects
        reward_maze = np.reshape(np.copy(self.rewards[:-1]), self.maze.shape)
        reward_maze[self.maze == 1] = 0
        #print(tuple(state2idcs(self.start_location, self.maze)))
        reward_maze[tuple(state2idcs(self.start_location, self.maze))] = np.NAN
        return np.array2string(reward_maze)

    def states_allowed_at_time(self, t):
        # all states are allowed, except wall states
        #return [s for s in range(self.n_states) if s not in self.wall_locations]
        return [s for s in range(self.n_states)]

    def actions_allowed_in_state(self, s):
        # you can not go inside walls and outside the grid
        allowed_actions = []
        i_coord, j_coord = state2idcs(s , self.maze)
        for action in ['up', 'down', 'right', 'left']:
            if action == 'up':  # up
                s1_idcs = [i_coord - 1, j_coord]
            elif action == 'down':  #  down
                s1_idcs = [i_coord + 1, j_coord]
            elif action == 'right':  #  right
                s1_idcs = [i_coord, j_coord + 1]
            else:  # left
                s1_idcs = [i_coord, j_coord - 1]
            s1_idcs = return_same_state_if_boundary([i_coord, j_coord], s1_idcs, self.maze)
            if not np.array_equal([i_coord, j_coord], s1_idcs):
                allowed_actions.append(action_str_to_num(action))
        return allowed_actions




