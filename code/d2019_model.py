#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:11:37 2020

@author: huanwang
"""

#https://pymc-devs.github.io/pymc/modelfitting.html
#https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb
#https://github.com/ricardoV94/stats/blob/master/modelling/RL_PyMC.ipynb
import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as tt

prob_intervention=0.3

data_file = '/Users/huanwang/GoogleDrive/Stanford/1st_yr/251/valence-control/data/exp1_data.csv'

dtf=pd.read_csv(data_file)

alpha_lr_prior=1
beta_lr_prior=1
alpha_prior=1
beta_prior=1
actions=dtf['subj_choice']-1
rewards=dtf['feedback']
conditons=dtf['condition']

inverse_temperature=
stickiness=
var_DKwhatthisis
#adapted from https://github.com/ricardoV94/stats/blob/master/modelling/RL_PyMC.ipynb

## to do: instead of using numeric priors, define probabilistic priors and use pymc to conduct the MAP.
## need to define a function to update learning rate for each trial and use this function to feed to theano converter codes before running pymc
def llik_td(x, *args):
    # Extract the arguments as they are passed by sp.optimize.minimize
    stickiness, inverse_temperature = x
    actions, rewards = args

    # Initialize values
    Q = np.array([.5, .5])
    log_prob_actions = np.zeros(len(actions))

    for t, (a, r, c) in enumerate(zip(actions,rewards,conditons)):
        
        
        
        # Apply the softmax transformation
        Q_ = Q * inverse_temperature + var_maybeChoice * stickiness
        var_maybeChoice=np.zeros((1,2))
        var_maybeChoice[0][a] = 1
        log_prob_action = Q_ - sp.special.logsumexp(Q_)

        # Store the log probability of the observed action
        log_prob_actions[t] = log_prob_action[a]
        
        #condition-specific definition of P(z|r,z), i.e. the degree to which feedback should be attributed to the intrinsic reward distribution rather than to the latent agent
        if c==2:
            if r==0:
                prob_attribution=1 #Benevolent agent (negative feedback)
            elif r==1:
                prob_attribution = (Q[a]*(1-prob_intervention)) / (Q[a]*(1-prob_intervention)+prob_intervention) #Benevolent agent (positive feedback)
        elif c==1:
            if r==0:
                prob_attribution = ((1-Q[a])*(1-prob_intervention))/((1-Q[a])*(1-prob_intervention)+prob_intervention) #Adversarial agent (negative feedback)
            elif r==1:
                prob_attribution=1 #Adversarial agent (positive feedback)
        elif c==3:
            if r==0:
                prob_attribution=((1-Q[a])*(1-prob_intervention))/((1-Q[a])*(1-prob_intervention)+prob_intervention/2) #Random agent (negative feedback)   
            elif r==1:
                prob_attribution=(Q[a]*(1-prob_intervention))/(Q[a]*(1-prob_intervention)+prob_intervention/2) #Random agent (positive feedback)
            
        #sum of past beliefs about latent agent non-intervention
        prob_attribution_cumulative[a] = prob_attribution_cumulative[a] + prob_attribution
        learning_rate = prob_attribution/(prob_attribution_cumulative[a] + alpha_lr_prior + beta_lr_prior)
        
        
        # Update the Q values for the next trial
        Q[a] = Q[a] + alpha * (r - Q[a])
        
        
        
    # Return the negative log likelihood of all observed actions
    return -np.sum(log_prob_actions[1:])
  

x0 = [alpha_prior, beta_prior]
result = sp.optimize.minimize(llik_td, x0, args=(actions, rewards), method='BFGS')
print(result)
print('')
print(f'MLE: alpha = {result.x[0]:.2f} (prior = {alpha_prior})')
print(f'MLE: beta = {result.x[1]:.2f} (prior = {beta_prior})')

llik_td([1,1],dtf['subj_choice']-1,dtf['feedback'])


# prob_attribution=pd.DataFrame(np.repeat(1,len(dtf)))
# with pm.Model() as model:
    
#     # empirical priors based on previous research (Gershman, 2016)
#     beta=pm.Gamma('beta',alpha=4.82,beta=0.88)
#     rho=pm.Normal('rho',mu=0.15,sigma=1.42)
#     # prior belief of true reward probability from participants
#     theta=pm.Beta('theta',alpha=1,beta=1)
    
#     #condition-specific definition of P(z|r,z), i.e. the degree to which feedback should be attributed to the intrinsic reward distribution rather than to the latent agent
# with model:
#     if dtf['condition']==2:
#         if dtf['feedback']==0:
#             prob_attribution=1 #Benevolent agent (negative feedback)
#         elif dtf['feedback']==1:
#             prob_attribution=(theta*(1-prob_intervention))/(theta*(1-prob_intervention)+prob_intervention) #Benevolent agent (positive feedback)
#     elif dtf['condition']==1:
#         if dtf['feedback']==0:
#             prob_attribution=((1-theta)*(1-prob_intervention))/((1-theta)*(1-prob_intervention)+prob_intervention) #Adversarial agent (negative feedback)
#         elif dtf['feedback']==1:
#             prob_attribution=1 #Adversarial agent (positive feedback)
#     elif dtf['condition']==3:
#         if dtf['feedback']==0:
#             prob_attribution=((1-theta)*(1-prob_intervention))/((1-theta)*(1-prob_intervention)+prob_intervention/2) #Random agent (negative feedback)   
#         elif dtf['feedback']==1:
#             prob_attribution=(theta*(1-prob_intervention))/(theta*(1-prob_intervention)+prob_intervention/2) #Random agent (positive feedback)


#     learning_rate=prob_attribution/(prob_attribution_cumulative+1+1)
    
#     Yobs=pm.Bernoulli('Yobs',p=,observed=Y)











# if dtf['condition']==2:
#     if dtf['feedback']==0:
#         prob_attribution[(dtf['condition']==2) & (dtf['feedback']==0)]=1 #Benevolent agent (negative feedback)
#     elif dtf['feedback']==1:
#         prob_attribution=(theta*(1-prob_intervention))/(theta*(1-prob_intervention)+prob_intervention) #Benevolent agent (positive feedback)
# elif dtf['condition']==1:
#     if dtf['feedback']==0:
#         prob_attribution[(dtf['condition']==1) & (dtf['feedback']==0)]=((1-theta)*(1-prob_intervention))/((1-theta)*(1-prob_intervention)+prob_intervention) #Adversarial agent (negative feedback)
#     elif dtf['feedback']==1:
#         prob_attribution[(dtf['condition']==1) & (dtf['feedback']==1)]=1 #Adversarial agent (positive feedback)
# elif dtf['condition']==3:
#     if dtf['feedback']==0:
#         prob_attribution[(dtf['condition']==3) & (dtf['feedback']==0)]=((1-theta)*(1-prob_intervention))/((1-theta)*(1-prob_intervention)+prob_intervention/2) #Random agent (negative feedback)   
#     elif dtf['feedback']==1:
#         prob_attribution[(dtf['condition']==3) & (dtf['feedback']==1)]=(theta*(1-prob_intervention))/(theta*(1-prob_intervention)+prob_intervention/2) #Random agent (positive feedback)





