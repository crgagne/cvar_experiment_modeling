import numpy as np
from mscl import calc_cvar_from_samples


def run_simulation(task,
                   policy,  # must be states x actions x alphas x time-steps
                   gamma,
                   T=3,
                   n_sims=10000,
                   s0=0,
                   alpha_i0=0,
                   alpha_set=None,
                   Xis=None,
                   adjust_alpha=False):

    # variables explained in run task
    total_rewards = []
    rewards = []
    states = []
    actions = []
    alphas = []
    sa_occup = []
    dis_sa_occup = []
    first_sa_occup = []
    most_prob_state = []
    most_prob_action = []
    action_slip_sa_occup = []
    state_slip_sa_occup = []
    slip_sa_occup = []

    # do n_sims rollouts
    for sim in range(n_sims):
        results = run_task(task, T, policy, gamma, s0, alpha_i0, alpha_set, Xis, adjust_alpha)
        # save results
        total_rewards.append(results['total_rewards'])
        states.append(results['states'])
        actions.append(results['actions'])
        rewards.append(results['rewards'])
        alphas.append(results['alphas'])
        sa_occup.append(results['sa_occup'])
        dis_sa_occup.append(results['dis_sa_occup'])
        first_sa_occup.append(results['first_sa_occup'])
        most_prob_state.append(results['most_prob_state'])
        most_prob_action.append(results['most_prob_action'])
        action_slip_sa_occup.append(results['action_slip_sa_occup'])
        state_slip_sa_occup.append(results['state_slip_sa_occup'])
        slip_sa_occup.append(results['slip_sa_occup'])

    results = {'returns': np.array(total_rewards), 'states': np.array(states), 'actions': np.array(actions),
               'rewards': np.array(rewards), 'alphas': np.array(alphas), 'sa_occup': np.array(sa_occup),
               'dis_sa_occup': np.array(dis_sa_occup), 'first_sa_occup': np.array(first_sa_occup),
               'most_prob_state': np.array(most_prob_state), 'most_prob_action': np.array(most_prob_action),
               'action_slip_sa_occup': np.array(action_slip_sa_occup),
               'state_slip_sa_occup': np.array(state_slip_sa_occup), 'slip_sa_occup': np.array(slip_sa_occup)}
    return results


def run_task(task,
             T,
             policy,
             gamma,
             s0,
             alpha0_i,
             alpha_set=None,
             Xis=None,
             adjust_alpha=False):


    rewards = []
    states = []
    actions = []
    alphas = []

    # shape: states x actions
    # state action occupancy, is incremented every time at [s, a] when action is chosen in state s
    sa_occup = np.zeros_like(policy[:, :, 0, 0])
    # same but discounted with gamma value
    discounted_sa_occup = np.zeros_like(policy[:, :, 0, 0])
    # only keeps track if state action pair occurred at least once, 1 for occurred, 0 otherwise
    first_sa_occup = np.zeros_like(policy[:, :, 0, 0])
    state_slip_sa_occup = np.zeros_like(policy[:, :, 0, 0])
    action_slip_sa_occup = np.zeros_like(policy[:, :, 0, 0])
    slip_sa_occup = np.zeros_like(policy[:, :, 0, 0])

    # track state slips, 0=slip, 1 = no slip
    most_prob_state = []
    # keeps track if the most probable action was chosen, 0=slip 1=no_slip
    most_prob_action = []
    # indicates whether best action was taken or the soft policy chose suboptimal action
    action_slip = False
    state_slip = False

    # set first state to start state
    s = s0
    # set alpha index to starting alpha index
    alpha_i = alpha0_i
    # corresponding alpha
    alpha = alpha_set[alpha_i]
    for t in range(T):

        # last action slip
        last_action_slip = action_slip

        # store state
        states.append(s)
        alphas.append(alpha)

        # choose action
        p_action = policy[s, :, alpha_i, t]
        # actions coded from 0 to 3, choose them with corresponding probabilities stored in policy
        a = np.random.choice(np.arange(len(p_action)), p=p_action)
        actions.append(a)

        # determine action slips
        action_slip = (a != np.argmax(p_action))
        most_prob_action.append(not action_slip)

        # state action occupancy
        sa_occup[s, a] += 1
        discounted_sa_occup[s, a] += gamma
        first_sa_occup[s, a] = 1

        if state_slip and not last_action_slip:  # took max action last time, but slipped states (current action could be anything)
            state_slip_sa_occup[s, a] = 1  # what is you current action taken in state
        if last_action_slip and not state_slip:  # didnt' take max action last time, didn't slip states
            action_slip_sa_occup[s, a] = 1
        if state_slip and last_action_slip:  # didn't take max action and slipped states
            slip_sa_occup[s, a] = 1

        # get reward for current state
        r = np.random.choice(task.r_support, p=task.p_rewards[s, :])
        r_i = np.where(task.r_support == r)[0][0]
        rewards.append(r)

        # get probabilities of all states when choosing action a in state s
        p = task.P[s, :, a]
        if np.sum(p) == 0:
            s_next = np.nan
        else:
            s_next = np.random.choice(np.arange(len(p)), p=p)

        state_slip = (s_next != np.argmax(p))
        most_prob_state.append(not state_slip)

        # adjust alpha using Xis if running a risk-dynamic policy
        # this is after action is chosen in state s according to original alpha level
        if adjust_alpha and not np.isnan(s_next):  # sp will be nan if transition probabilities are all 0
            xi = Xis[s, a, alpha_i, r_i, s_next, t]
            alpha = xi * alpha
            alpha_i = np.argmin(np.abs(alpha_set - alpha))
            alpha = alpha_set[alpha_i]

        # set current state to next state
        s = s_next

    total_rewards = np.sum(rewards)
    results = {'total_rewards': total_rewards, 'rewards': rewards, 'states': states, 'actions': actions,
               'alphas': alphas, 'dis_sa_occup': discounted_sa_occup, 'sa_occup': sa_occup,
               'first_sa_occup': first_sa_occup, 'most_prob_state': most_prob_state,
               'most_prob_action': most_prob_action, 'action_slip_sa_occup': action_slip_sa_occup,
               'state_slip_sa_occup': state_slip_sa_occup, 'slip_sa_occup': slip_sa_occup}
    return results


def calc_V_CVaR_MCMC(alpha_set, returns):
    V_CVaR_MCMC = []
    for alpha in alpha_set:
        if alpha == 0.0:
            cvar = np.min(returns)
        else:
            var, cvar = calc_cvar_from_samples(returns, alpha)
        V_CVaR_MCMC.append(cvar)

    return (np.array(V_CVaR_MCMC))
