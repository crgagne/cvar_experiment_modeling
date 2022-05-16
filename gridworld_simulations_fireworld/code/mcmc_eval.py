import numpy as np
from mscl import calc_cvar_from_samples

def run_simulation(task,
                   policy,  # must be states x actions x alphas x time-steps
                   gamma,
                   T = 3,
                   Nsims = 10000,
                   s0 = 0,
                   alpha_i0= 0,
                   alpha_set = None,
                   Xis=None,
                   adjust_alpha=False):

    returns = []
    rewards = []
    states = []
    actions  = []
    alphas = []
    sa_occup = []
    dis_sa_occup = []
    first_sa_occup = []
    most_prob_state = []
    most_prob_action = []
    action_slip_sa_occup = []
    state_slip_sa_occup = []
    slip_sa_occup = []
    for sim in range(Nsims):
        results = run_task(task,T,policy,gamma,s0,alpha_i0,alpha_set,Xis,adjust_alpha)
        returns.append(results['R'])
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

    results = {}
    results['returns']=np.array(returns)
    results['states']=np.array(states)
    results['actions']=np.array(actions)
    results['rewards']=np.array(rewards)
    results['alphas']=np.array(alphas)
    results['sa_occup']=np.array(sa_occup)
    results['dis_sa_occup']=np.array(dis_sa_occup)
    results['first_sa_occup']=np.array(first_sa_occup)
    results['most_prob_state']=np.array(most_prob_state)
    results['most_prob_action']=np.array(most_prob_action)
    results['action_slip_sa_occup']=np.array(action_slip_sa_occup)
    results['state_slip_sa_occup']=np.array(state_slip_sa_occup)
    results['slip_sa_occup']=np.array(slip_sa_occup)
    return(results)


def run_task(task,
             T,
             policy,
             gamma,
             s0=0,
             alpha0_i= 0,
             alpha_set = None,
             Xis = None,
             adjust_alpha=False):

    s=s0
    alpha_i = alpha0_i
    alpha = alpha_set[alpha_i]
    rewards = []
    states = []
    actions  = []
    alphas = []
    sa_occup = np.zeros_like(policy[:,:,0,0]) # only do sa to save spce
    dis_sa_occup = np.zeros_like(policy[:,:,0,0])
    first_sa_occup = np.zeros_like(policy[:,:,0,0])
    state_slip_sa_occup = np.zeros_like(policy[:,:,0,0])
    action_slip_sa_occup = np.zeros_like(policy[:,:,0,0])
    slip_sa_occup = np.zeros_like(policy[:,:,0,0])
    most_prob_state = []
    most_prob_action = []
    action_slip = False
    state_slip = False
    #import pdb; pdb.set_trace()

    for t in range(T):

        # last action slip
        last_action_slip = action_slip

        # store state
        states.append(s)
        alphas.append(alpha)

        # choose action
        #try:
        p_action = policy[s,:,alpha_i,t]
        #except:
        #    import pdb; pdb.set_trace()
        a = np.random.choice([i for i in range(len(p_action))],p=p_action)
        actions.append(a)

        # action slips
        if a==np.argmax(p_action):
            action_slip=False
            most_prob_action.append(1)
        else:
            action_slip=True
            most_prob_action.append(0)

        # state action occupancy
        sa_occup[s,a]+=1
        dis_sa_occup[s,a]+=gamma
        first_sa_occup[s,a]=1 # only set it for the first time

        if state_slip and not last_action_slip: # took max action last time, but slipped states (current action could be anything)
            state_slip_sa_occup[s,a]=1 # what is you current action taken in state
        if last_action_slip and not state_slip: # didnt' take max action last time, didn't slip states
            action_slip_sa_occup[s,a]=1
        if state_slip and last_action_slip: # didn't take max action and slipped states
            slip_sa_occup[s,a]=1

        # get reward for current state
        r = np.random.choice(task.r_support,p=task.p_rewards[s,:])
        r_i = np.where(task.r_support==r)[0][0]
        rewards.append(r)

        # get next state
        p = task.P[s,:,a]
        if np.sum(p)==0:
            sp = np.nan
        else:
            sp = np.random.choice(np.arange(len(p)),p=p)

        if sp==np.argmax(p):
            state_slip=False
            most_prob_state.append(1)
        else:
            state_slip=True
            most_prob_state.append(0)

        # adjust alpha using Xis if running a risk-dynamic policy
        # this is after action is chosen in state s according to original alpha level
        if adjust_alpha and not np.isnan(sp): # sp will be nan if transition probabilities are all 0
            xi = Xis[s,a,alpha_i,r_i,sp,t]
            alpha = xi*alpha
            alpha_i = np.argmin(np.abs(alpha_set-alpha))
            alpha = alpha_set[alpha_i]

        # set current state to next state
        s = sp

    R=np.sum(rewards)
    results = {}
    results['R']=R
    results['rewards']=rewards
    results['states']=states
    results['actions']=actions
    results['alphas']=alphas
    results['dis_sa_occup']=dis_sa_occup
    results['sa_occup']=sa_occup
    results['first_sa_occup']=first_sa_occup
    results['most_prob_state']=most_prob_state
    results['most_prob_action']=most_prob_action
    results['action_slip_sa_occup']=action_slip_sa_occup
    results['state_slip_sa_occup']=state_slip_sa_occup
    results['slip_sa_occup']=slip_sa_occup
    return(results)


def calc_V_CVaR_MCMC(alpha_set,returns):

    V_CVaR_MCMC = []
    for alpha in alpha_set:
        if alpha==0.0:
            cvar=np.min(returns)
        else:
            var,cvar = calc_cvar_from_samples(returns,alpha)
        V_CVaR_MCMC.append(cvar)

    return(np.array(V_CVaR_MCMC))
