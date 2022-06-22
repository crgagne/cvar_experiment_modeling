import numpy as np

def action_str_to_num(a):

    if a == 'up':
        return(0)
    elif a == 'down':
        return(1)
    elif a == 'right':
        return(2)
    elif a == 'left':
        return(3)

def action_num_to_str(a):

    if a == 0:
        return('up')
    elif a == 1:
        return('down')
    elif a == 2:
        return('right')
    elif a == 3:
        return('left')

def idcs2state(idcs: list, maze, order='C'):
    '''Convert state idcs to state id
    Notes:
       order='C' is row major [[1,2,3],[4,5,6]]
       order= is column major [[1,3,4],[2,4,6]]

    Example:
        maze = np.zeros((3,4))
        for s in range(Ns_wo_absorb):
            idx = state2idcs(s,maze,order=)
            s_rec = idcs2state(idx,maze,order=)
            maze[idx[0],idx[1]]=s_rec
        print(maze)

    '''
    si = idcs[0]
    sj = idcs[1]
    side_j = maze.shape[1]
    side_i = maze.shape[0]

    if order =='C':
        return si*side_j + sj
    if order =='F':
        return sj*side_i + si


def state2idcs(s: int, maze, state_mult=1,order='C'):
    '''Convert state id to state idcs
    '''

    # convert state to location id
    num_locs = maze.shape[0]*maze.shape[1]
    loc = s%(num_locs)

    # convert location id to i,j coordinates
    if order =='C':

        side_j = maze.shape[1]
        si = loc // side_j
        sj = loc % side_j

    elif order=='F':

        side_j = maze.shape[0] # swapped
        sj = loc // side_j # swapped
        si = loc % side_j

    return [int(si), int(sj)]


def return_same_state_if_boundary(s_idcs,s1_idcs,maze):
    # if transition outside the maze, return current state
    if (s1_idcs[0]>=maze.shape[0]) \
        or (s1_idcs[0]<0) \
        or (s1_idcs[1]>=maze.shape[1]) \
        or (s1_idcs[1]<0):
        s1_idcs = s_idcs

    # if wall, return current state
    if maze[s1_idcs[0],s1_idcs[1]]==1:
        s1_idcs = s_idcs

    return(s1_idcs)

def get_next_state_prob_from_maze(s_idcs,
                                    a,
                                    maze,
                                    noise_mode='none',
                                    err_prob=0,
                                    order='C'):
    '''Returns a vector of probabilities for next states transitions
    '''

    # current coordinates
    i_coord = s_idcs[0]
    j_coord = s_idcs[1]
    s1_idcs = []

    # place holder for probs
    num_locs = maze.shape[0]*maze.shape[1]
    s1_probs = np.zeros(num_locs)

    # current state number
    s = idcs2state(s_idcs,maze,order=order)

    if noise_mode=='none':

        # adjust coordinates
        if a == 0: # up
            s1_idcs = [i_coord-1, j_coord]
        elif a == 1: # down
            s1_idcs = [i_coord+1, j_coord]
        elif a == 2: # right
            s1_idcs = [i_coord, j_coord+1]
        else:      # left
            s1_idcs = [i_coord, j_coord-1]

        # if out of bounds, return current index
        s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)

        # get the state number for new index
        s1 = idcs2state(s1_idcs, maze, order=order)

        # set probability to 1
        s1_probs[s1]=1.0

    if noise_mode=='adjacent_noisy':
        if a == 0: # up
            s1_idcs = [i_coord-1, j_coord] # up
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=(1-err_prob)

            s1_idcs = [i_coord+1, j_coord] # down
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(1/5)

            s1_idcs = [i_coord, j_coord+1] # right
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)

            s1_idcs = [i_coord, j_coord-1] # left
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)
        elif a == 1: # down
            s1_idcs = [i_coord-1, j_coord] # up
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(1/5)

            s1_idcs = [i_coord+1, j_coord] # down
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=(1-err_prob)

            s1_idcs = [i_coord, j_coord+1] # right
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)

            s1_idcs = [i_coord, j_coord-1] # left
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)
        elif a == 2: # right
            s1_idcs = [i_coord-1, j_coord] # up
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)

            s1_idcs = [i_coord+1, j_coord] # down
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)

            s1_idcs = [i_coord, j_coord+1] # right
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=(1-err_prob)

            s1_idcs = [i_coord, j_coord-1] # left
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(1/5)
        elif a == 3: # left
            s1_idcs = [i_coord-1, j_coord] # up
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)

            s1_idcs = [i_coord+1, j_coord] # down
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(2/5)

            s1_idcs = [i_coord, j_coord+1] # right
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=err_prob*(1/5)

            s1_idcs = [i_coord, j_coord-1] # left
            s1_idcs = return_same_state_if_boundary(s_idcs,s1_idcs,maze)
            s1 = idcs2state(s1_idcs, maze, order=order)
            s1_probs[s1]+=(1-err_prob)

    return(s1_probs)


def build_P_from_maze(maze, n_states, n_actions, noise_mode, err_prob,
                      absorbing_states, terminal_state, order='C',
                      add_termin_up=False, up_termin_prob=0.05):

    P = np.zeros((n_states, n_states, n_actions))

    for s in range(n_states):

        # get state index
        s_idcs = state2idcs(s, maze, order=order)

        for a in range(n_actions):

            # get next state probabilities
            s1_prob = get_next_state_prob_from_maze(s_idcs,a,maze,noise_mode,err_prob,order=order)

            # deal with aborbing states; probability = [0,0,0,..,1]; terminal state is always last state
            if s in absorbing_states or s == terminal_state:
                s1_prob = np.zeros(n_states)
                s1_prob[-1] = 1

            # add probs
            else:
                if len(absorbing_states)>=1:
                    s1_prob = np.append(s1_prob,np.array([0])) # append terminal state

            if add_termin_up and a==0 and s not in absorbing_states and s!=terminal_state:
                s1_prob = s1_prob*(1-up_termin_prob)
                s1_prob[-1]=up_termin_prob
                if s1_prob.sum()!=1.0:
                    import pdb; pdb.set_trace()

            P[s,:,a]=s1_prob

    return(P)
