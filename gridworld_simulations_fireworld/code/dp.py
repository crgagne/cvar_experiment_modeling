import numpy as np
import itertools
from scipy.optimize import minimize

import multiprocessing
from multiprocessing import Pool

global break_early1
global break_early2
global total
break_early1=0
break_early2=0
total=0


def CVaR_DP(task,
            T=3,
            alpha0=0.3,
            alpha_set = np.array([0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.0]),
            cvar_type='pCVaR',
            gamma=1.0,
            interp_type='V',
            policy_to_evaluate=None,
            Q_roundoff=4,
            verbose=True,
            mode='fast',
            parallel=False):


    if mode=='fast':
        # alpha_set = np.array([0,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.0])
        same_answer_ns=1

    else:
        # alpha_set = np.array([0.        , 0.01      , 0.01274275, 0.01623777, 0.02069138,
        #                        0.02636651, 0.03359818, 0.04281332, 0.05455595, 0.06951928,
        #                        0.08858668, 0.11288379, 0.14384499, 0.18329807, 0.23357215,
        #                        0.29763514, 0.37926902, 0.48329302, 0.61584821, 0.78475997,
        #                        1.        ])
        same_answer_ns=3


    n_states = task.n_states
    n_actions = task.n_actions
    Nalpha = len(alpha_set)
    n_rewards = task.n_rewards

    alpha0_i = np.where(alpha_set==alpha0)[0]

    # q values at for every state action pair for each timestep and alpha
    Q_CVaR = np.zeros((n_states,n_actions,Nalpha,T))
    # what is this for?
    Xis = np.zeros((n_states,n_actions,Nalpha,n_rewards,n_states,T))
    # same for statevalues
    V_CVaR = np.zeros((n_states,Nalpha,T))

    pi = np.zeros((n_states,n_actions,Nalpha,T))

    ######################################
    # work backwards from last time step #
    ######################################

    for t in reversed(range(0,T)):

        if verbose:
            print('t='+str(t))

        ##################################
        ## Update State,Action Q-Values ##
        #################################


        # these are just all the states
        states_to_iterate = task.states_allowed_at_time(t)

        # loop over state space
        if parallel:
            with Pool(multiprocessing.cpu_count()) as p:
                map_results = p.starmap(Update_Q_Values, [(s,t,V_CVaR,Nalpha,n_actions,n_rewards,
                  n_states,T,alpha_set,task,cvar_type,gamma,interp_type,same_answer_ns) for s in states_to_iterate] )
        else:
            map_results = []
            for s in states_to_iterate:
                Q_CVaR_tmp,Xis_tmp =  Update_Q_Values(s,t,V_CVaR,Nalpha,n_actions,n_rewards,
                                        n_states,T,alpha_set,task,cvar_type,gamma,interp_type,same_answer_ns)
                map_results.append((Q_CVaR_tmp,Xis_tmp))

        # unpack
        for s in states_to_iterate:
            Q_CVaR[s,:,:,t] = map_results[s][0]
            Xis[s,:,:,:,:,t] = map_results[s][1]

        ####################################
        ## Update State Values and Policy ##
        ####################################

        # loop over state space
        states_to_iterate = task.states_allowed_at_time(t)
        for s in states_to_iterate: # I could also maybe just do states to consider each round..

            # loop over possible alphas
            alphas_to_iterate = range(Nalpha)
            for alpha_i in alphas_to_iterate:

                alpha = alpha_set[alpha_i]



                # fcvar fix
                # do a loop in here for all alpha0 values
                # use that to upgrade state value of fCVAR
                #
                ## update policy ##
                if policy_to_evaluate is None:
                    if cvar_type=='fCVaR':

                        actions_allowed = np.array(task.actions_allowed_in_state(s))

                        # No matter what alpha level you are in the loop, use the alpha0_i to determine policy
                        Q_CVaR[s,:,alpha0_i,t] = np.round(Q_CVaR[s,:,alpha0_i,t],Q_roundoff)
                        Q_best_alpha0 = np.max(Q_CVaR[s,actions_allowed,alpha0_i,t])
                        best_actions = np.where(np.squeeze(Q_CVaR[s,:,alpha0_i,t])==Q_best_alpha0)[0]

                        filter=np.isin(best_actions,actions_allowed)
                        best_actions = best_actions[filter]

                        # (implicit tie-breaker to choose the first option)
                        # usually its a single action anyway
                        best_action = best_actions[0]

                        # create policy for current alpha level in loop
                        pi[s,:,alpha_i,t]=0.0
                        pi[s,best_action,alpha_i,t]=1.0

                        # update CVaR state value using state,action value using current alpha in loop (transfering distribution from chosen action)
                        Q_best = Q_CVaR[s,best_action,alpha_i,t]
                        V_CVaR[s,alpha_i,t] = Q_best


                    elif cvar_type=='pCVaR' or cvar_type=='nCVaR':

                        # ADDING: NOT 100% Vetted
                        actions_allowed = np.array(task.actions_allowed_in_state(s))

                        # otherwise use alpha_i
                        Q_CVaR[s,:,alpha_i,t] = np.round(Q_CVaR[s,:,alpha_i,t],Q_roundoff) # round Q-values so that 'numerical ties' are obvious; not necessary but cleaner for looking at policy
                        Q_best = np.nanmax(Q_CVaR[s,actions_allowed,alpha_i,t])
                        best_actions = np.where(np.squeeze(Q_CVaR[s,:,alpha_i,t])==Q_best)[0]

                        filter=np.isin(best_actions,actions_allowed)
                        best_actions = best_actions[filter]

                        # (implicit tie-breaker to choose the first option)
                        # usually its a single action anyway
                        best_action = best_actions[0]

                        # create policy
                        pi[s,:,alpha_i,t]=0.0
                        pi[s,best_action,alpha_i,t]=1.0

                        ## update CVaR value
                        V_CVaR[s,alpha_i,t] = Q_best

                ## evaluate existing policy ##
                else:
                    # improper for stochastic policies
                    V_CVaR[s,alpha_i,t] =  np.sum(policy_to_evaluate[s,:,alpha_i,t]*Q_CVaR[s,:,alpha_i,t])

    output = {}
    output['Q_CVaR']=Q_CVaR
    output['pi']=pi
    output['V_CVaR']=V_CVaR
    output['Xis']=Xis

    print(total)
    print(break_early1)
    print(break_early2)

    return output


def interpolate_V(V,state_next,alpha_next,alpha_set,t,debug=False):
    '''Interpolates CVaR value function between alpha levels.
       Returns V(s,alpha)
    '''

    assert alpha_set[0]==0.0 # make sure 0 is part of interpolation

    alpha_next_i_nearest = np.argmin(np.abs(alpha_set-alpha_next))
    alpha_next_nearest = alpha_set[alpha_next_i_nearest]

    if alpha_next>1:
        # return highest alpha
        return(V[state_next,len(alpha_set)-1,t])
    elif alpha_next<0:
        # shouldn't be able to happen, so break if it does
        import pdb; pdb.set_trace()
    elif alpha_next==alpha_next_nearest:
        # no need for interpolation.
        return(V[state_next,alpha_next_i_nearest,t])
    else:
        # find lower and upper y_nearest.
        if alpha_next_nearest<alpha_next:
            alpha_next_i_upper = alpha_next_i_nearest+1
            alpha_next_upper = alpha_set[alpha_next_i_upper]
            alpha_next_i_lower = alpha_next_i_nearest
            alpha_next_lower = alpha_next_nearest
        elif alpha_next_nearest>alpha_next:
            alpha_next_i_upper = alpha_next_i_nearest
            alpha_next_upper = alpha_next_nearest
            alpha_next_i_lower = alpha_next_i_nearest-1
            alpha_next_lower = alpha_set[alpha_next_i_lower]

        # Slope
        slope = (V[state_next,alpha_next_i_upper,t] - V[state_next,alpha_next_i_lower,t]) / (alpha_next_upper - alpha_next_lower)

        # Start at lower and difference times the slope
        V_interp = V[state_next,alpha_next_i_lower,t] + slope*(alpha_next-alpha_next_lower)

        return(V_interp)

def distorted_value_objective_fun(dist_weights,
                                  next_state_reward_pairs,
                                  prob_next_state_reward_pairs,
                                  V,
                                  alpha,
                                  alpha_i,
                                  alpha_set,
                                  t,
                                  gamma,
                                  cvar_type='pCVaR',
                                  debug=False,
                                  interp_type='V'):

    if np.any(np.isnan(dist_weights)):
        return(np.inf)

    distorted_exp=0.0

    # loop over next states and rewards, associated probabilities, and distortion weights
    for (next_state,reward),prob,dweight in zip(next_state_reward_pairs,
                                                prob_next_state_reward_pairs,
                                                dist_weights):

        # calculate interpolate value function
        if cvar_type=='pCVaR' or cvar_type=='fCVaR':
            Vp=interpolate_V(V,next_state,alpha*dweight,alpha_set,t+1)
            distorted_exp += prob*dweight*(reward + gamma*Vp) # no multiplication by distortion weight
        elif cvar_type=='nCVaR':
            Vp=V[next_state,alpha_i,t+1]
            distorted_exp += prob*dweight*(reward + gamma*Vp)

    return(distorted_exp)


def Q_backup(next_states,
            p_next_states,
            rewards,
            p_rewards,
            alpha,
            alpha_i,
            V,
            t, # time-step
            cvar_type,
            gamma,
            alpha_set,
            max_inner_iters=50,
            multi_starts_N=10,
            same_answer_ns=3,
            same_answer_tol=1e-4,
            interp_type='V',
            ):

        '''
        Q(s,a,alpha,t) <- min xi sum_(r,s') p(s'|s,a)r(r|s,a)xi(r,s')[r + gammaV(s',alpha*xi(r,s'),t+t1)]

        '''

        # set up next state reward pairs
        next_state_reward_pairs = [pair for pair in itertools.product(next_states,rewards)]
        prob_next_state_reward_pairs = [probs[0]*probs[1] for probs in itertools.product(p_next_states,p_rewards)] # indedendent so multiply

        # handle alpha=0
        if alpha == 0:

            # find minimum next state value
            minV = np.min(V[next_states,0,t+1])
            #minV_idcs = np.where(V[next_states,0,t+1]==minV)[0]

            # find the minimum reward
            minr = np.min(rewards)
            #minr_idcs = np.where(rewards==minr)[0]

            # calculate current value
            Q = minr + gamma*minV

            # make corresponding weights
            dist_weights = np.zeros((len(rewards),len(next_states)))*np.nan

            # assert succes
            success = True

            return(Q,dist_weights,success)

        else:

            # distortion weight bounds
            bnds = tuple(((0.0,1.0/alpha) for _ in range(len(prob_next_state_reward_pairs))))

            # sum to 1 constraint for distortion weights
            def sum_to_1_constraint(dist_weights):
                zero = np.dot(dist_weights,np.array(prob_next_state_reward_pairs))-1 # p_trans is found in one python env up
                return(zero)

            cons = ({'type': 'eq', 'fun': sum_to_1_constraint})
            succeeded_at_least_once = False
            results_list = []
            fun_mins = []

            global total
            total+=1
            for _ in range(max_inner_iters):

                # distortion weight initial values
                n_probs = len(prob_next_state_reward_pairs)

                # uniform
                #dist_weights_init = np.random.uniform(alpha_set[1],1.0/alpha,n_probs)

                # uniform in probability simplex
                dist_probs_init = np.random.dirichlet(np.ones(n_probs),size=1)[0]
                dist_weights_init = dist_probs_init/np.array(prob_next_state_reward_pairs)
                assert np.abs(sum_to_1_constraint(dist_weights_init))<1e-4

                results = minimize(distorted_value_objective_fun,
                                   dist_weights_init,
                                   args=(next_state_reward_pairs,prob_next_state_reward_pairs,
                                   V,alpha,alpha_i,alpha_set,t,gamma,cvar_type,False,interp_type),
                                   method='SLSQP',
                                   bounds=bnds,
                                   constraints=cons)

                # figure out whether scipy thinks it succeeded
                if results.success:
                    succeeded_at_least_once=True
                    results_list.append(results)
                    fun_mins.append(results.fun)

                # exit early if all N minimums are the same; N specified.
                if len(fun_mins)>same_answer_ns:
                    minn = np.min(np.array(fun_mins))
                    num_within_tol = np.sum((np.array(fun_mins)-minn)<same_answer_tol)
                    if num_within_tol>=same_answer_ns:
                        global break_early1
                        break_early1+=1
                        break

                # exit after number of multi-starts have been exceeded.
                if len(fun_mins)>multi_starts_N:
                    global break_early2
                    break_early2+=1
                    break

            # find minimum over multi-starts
            argmin_funs = np.argmin(np.array(fun_mins))
            results = results_list[argmin_funs]

            # unpack results
            dist_weights = results.x
            Q = results.fun
            success = results.success
            if success==False:
                print('Failed')

            return(Q,dist_weights,success)


def Q_backup_horizon(rewards,
                     p_rewards,
                     alpha,
                     alpha_set):

    assert type(rewards)==np.ndarray
    assert len(rewards)==len(p_rewards)

    if alpha==0.0:
        #print(rewards)
        minR = np.min(rewards)
        minR_idcs = np.where(rewards==minR)[0]

        Q = minR
        dist_weights = np.zeros(len(p_rewards))
        dist_weights[minR_idcs]=1/(p_rewards[minR_idcs]*len(minR_idcs)) # take the weight as 1/prob, but if there are more than 1 additionally divide by total)

        assert np.abs((dist_weights*p_rewards).sum()-1)<0.01

        success=True
        return(Q,dist_weights,success)

    else:

        def obj_fun(dist_weights,p_rewards,rewards):
            return(np.sum((dist_weights*p_rewards*rewards)))

        def sum_to_1_constraint(dist_weights):
            return(np.dot(dist_weights,p_rewards)-1)

        bnds = tuple(((0.0,1.0/alpha) for _ in range(len(p_rewards))))
        dist_weights_init = np.random.uniform(alpha_set[1],1.0/alpha,len(rewards))
        cons = ({'type': 'eq', 'fun': sum_to_1_constraint})

        results = minimize(obj_fun,
                           dist_weights_init,
                           args=(p_rewards,rewards),
                           method='SLSQP',
                           bounds=bnds,
                           constraints=cons)

        dist_weights = results.x
        Q = results.fun
        success = results.success

        assert np.abs((1-np.dot(dist_weights,p_rewards)))<0.01
        assert np.all(dist_weights<=(1.0/alpha+0.01))
        assert np.all(0<=dist_weights)

        return(Q, dist_weights, success)


def Update_Q_Values(s,
                    t,
                    V_CVaR,
                    Nalpha,
                    n_actions,
                    n_rewards,
                    n_states,
                    T,
                    alpha_set,
                    task,
                    cvar_type,
                    gamma,
                    interp_type,
                    same_answer_ns):
    '''Update for all actions and alphas at a single state'''



    Q_CVaR_tmp = np.zeros((n_actions, Nalpha))
    Q_CVaR_tmp[:, :] = np.nan

    Xis_tmp = np.zeros((n_actions, Nalpha, n_rewards, n_states))

    # loop over possible alphas
    alphas_to_iterate = range(Nalpha)
    #print('update Q values at step: {}, state{}'.format(t,s))
    for alpha_i in alphas_to_iterate:


        alpha = alpha_set[alpha_i] # get alpha

        #if s==5 and t<13 and alpha==0.11288379:
        #    import pdb; pdb.set_trace()

        # loop over actions

        # Q: what does this except statement want to catch
        try:
            actions_to_iterate = task.actions_allowed_in_state(s)

        except:
            actions_to_iterate = range(n_actions)

        for a in actions_to_iterate:

            ## update CVaR Q-value
            # at horizon
            if t==(T-1):
                #
                # if s==6 and alpha==0.11288379:
                #    import pdb; pdb.set_trace()
                # get possible rewards current state
                non_zero_reward_idcs = np.where(task.p_rewards[s,:]!=0.0)[0] # where probability is not zero

                rewards = task.r_support[non_zero_reward_idcs]
                p_rewards = task.p_rewards[s,non_zero_reward_idcs]


                Q_CVaR_tmp[a,alpha_i],xis,success = Q_backup_horizon(np.array(rewards),
                                                         np.array(p_rewards),
                                                         alpha,
                                                         alpha_set)

                Xis_tmp[a,alpha_i,non_zero_reward_idcs,:]=np.tile(xis[:,np.newaxis], n_states)

            else:

                # get possible rewards current state
                non_zero_reward_idcs = np.where(task.p_rewards[s,:]!=0.0)[0] # where probability is not zero
                rewards = task.r_support[non_zero_reward_idcs]
                p_rewards = task.p_rewards[s,non_zero_reward_idcs]

                # get next states with non-zero transition prob
                next_states = np.where(task.P[s, :, a] != 0.0)[0]
                p_next_states = task.P[s, next_states, a]

                #import pdb; pdb.set_trace()

                if len(next_states)==0:
                    import pdb; pdb.set_trace()

                # do Q-value back-up
                Q_CVaR_tmp[a,alpha_i],xis,success = Q_backup(next_states,
                                                             p_next_states,
                                                             rewards,
                                                             p_rewards,
                                                             alpha,
                                                             alpha_i,
                                                             V_CVaR,
                                                             t, # time-step
                                                             cvar_type,
                                                             gamma,
                                                             alpha_set,
                                                             interp_type=interp_type,
                                                             same_answer_ns=same_answer_ns)

                # store Xis over next states and possible rewards
                try:
                    Xis_tmp[a,alpha_i,non_zero_reward_idcs,next_states]=np.squeeze(xis)
                except:
                    Xis_tmp[a,alpha_i,non_zero_reward_idcs,next_states]=np.nan

            #print(Q_CVaR_tmp)

    return Q_CVaR_tmp,Xis_tmp


