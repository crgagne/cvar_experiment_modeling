import numpy as np
from scipy.stats import norm
from task_utils import state2idcs, build_P_from_maze

class Task_Lava_Lake:

    # Example Diagram 5x5#
    #  [_, _, _, _, _]
    #  [_, _, _, _, _]
    #  [S, _, L, _, G]
    #  [_, _, _, _, _]
    #  [_, _, _, _, _]

    def __init__(self,nrows=5,
                      ncols=5,
                      start_location=10,
                      lava_locations=[12],
                      goal_locations=[14],
                      wall_locations=[],
                      err_prob=0.05,
                      noise_mode='adjacent_noisy',
                      goal_reward=3,
                      lava_reward=-10,
                      order='C'):

        # params
        self.Ns = nrows*ncols + 1 # with terminal state
        self.maze = np.zeros((nrows,ncols))
        self.Na = 4
        self.err_prob=err_prob
        self.noise_mode=noise_mode
        self.start_location=start_location
        self.lava_locations=lava_locations
        self.goal_locations=goal_locations
        self.wall_locations=wall_locations
        self.terminal_state = self.Ns-1
        self.absorbing_states = self.lava_locations+self.goal_locations
        self.goal_reward = goal_reward
        self.lava_reward = lava_reward
        self.Rmin = lava_reward
        self.Rmax = goal_reward
        self.dr = 1
        self.reward_dicts  = {}
        for loc in self.goal_locations:
            self.reward_dicts[loc]=self.goal_reward
        for loc in self.lava_locations:
            self.reward_dicts[loc]=self.lava_reward

        # add maze with state labels
        self.maze_w_state_labels = self.maze.copy()
        for state in range(self.Ns-1):
            idcs = state2idcs(state,self.maze,order=order)
            self.maze_w_state_labels[idcs[0],idcs[1]]=state

        # add wall states to maze
        for s in self.wall_locations:
            idcs = state2idcs(s,self.maze,order=order)
            self.maze[idcs[0],idcs[1]]=1

        # state transition matrix ##
        self.P = build_P_from_maze(self.maze,
                      self.Ns,
                      self.Na,
                      self.noise_mode,
                      self.err_prob,
                      self.absorbing_states,
                      self.terminal_state,
                      order=order)

        # reward range and possible values
        self.r_support = np.arange(self.Rmin,self.Rmax+self.dr,self.dr) # [-1,0,1,2,3]
        self.Nr = len(self.r_support)

        # create reward distributions per state (or deterministc in this case),
        # and a list of rewards associated with each state
        self.p_r = np.zeros((self.Ns,self.Nr))
        self.rewards = np.zeros(self.Ns)

        for state in range(self.Ns):
            if state in self.reward_dicts.keys():
                r =self.reward_dicts[state]
            else:
                r = 0 # fill in all non-rewarded states with reward = 0
            self.rewards[state]=r
            self.p_r[state,np.where(self.r_support==r)]=1.0 # deterministic rewards

    def states_allowed_at_time(self,t):
        return([s for s in range(self.Ns)])

    def actions_allowed_in_state(self,s):
        allowed = [0,1,2,3]
        return allowed