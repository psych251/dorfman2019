#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 07:20:13 2020

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
import theano
import seaborn as sns
from matplotlib.lines import Line2D
import statsmodels.api as sm
from statsmodels.formula.api import ols


prob_intervention=0.3 # Different from the authors matlab code, it is 0.5 presuambly representing no bias towards either options. It is inconsistent with the paper which has written that they have used the true probability of 0.3 which was part of the instruction to the participants.

datadir='/Users/huanwang/GoogleDrive/Stanford/1st_yr/251/valence-control/data/'

data_file = datadir+'exp1_data.csv'

dtf_all=pd.read_csv(data_file)

#loop through all subjects and fit model at the individual level
dt_Grand_p_attribution=pd.DataFrame()
dt_Grand_learning_rate=pd.DataFrame()

for isubj in range(len(dtf_all.subject.unique())):
    
    alpha_lr_prior=0
    beta_lr_prior=0
    alpha_prior=1
    beta_prior=1
    dtf=dtf_all[dtf_all['subject']==dtf_all.subject.unique()[isubj]]
    actions=dtf['subj_choice']-1
    rewards=dtf['feedback']
    conditions=dtf['condition']
    blocks=dtf['block_num']
    
    
    #### Estimate the paramters using Maximum likelihood estimation without Bayesian priors (i.e. the inverse_temperature and stickiness)
    
    def llik_td(x, *args):
        # Extract the arguments as they are passed by sp.optimize.minimize
        inverse_temperature, stickiness = x
        actions, rewards, conditions, blocks = args
    
        # Initialize values
        #Q = np.array([.5, .5])
        log_prob_actions = np.zeros((len(actions),2))
    
        for i_t, (i_a, i_r, i_c, i_b) in enumerate(zip(actions,rewards,conditions,blocks)):
            
            # redefine parameters to intial values at first trial of each block
            #### still having some problems with this if statements should really only happen for 3 times once per block
            if i_t==0:
                var_maybeChoice=np.zeros((1,2))
                Q = np.array([.5, .5])
                prob_attribution_cumulative=np.array([alpha_lr_prior+beta_lr_prior,alpha_lr_prior+beta_lr_prior],dtype="float64")
            elif i_t>0 and i_b!=blocks.iloc[i_t-1]:
                var_maybeChoice=np.zeros((1,2))
                Q = np.array([.5, .5])
                prob_attribution_cumulative=np.array([alpha_lr_prior+beta_lr_prior,alpha_lr_prior+beta_lr_prior],dtype="float64")
    
            # Apply the softmax transformation
            Q_ = Q * inverse_temperature + var_maybeChoice * stickiness
            var_maybeChoice=np.zeros((1,2))
            var_maybeChoice[0][i_a] = 1
            log_prob_action = Q_ - sp.special.logsumexp(Q_)
    
            # Store the log probability of the observed action
            # which I think is redundant and can be stored without creating a new var but anyway....
            log_prob_actions[i_t] = log_prob_action
            
            #condition-specific definition of P(z|r,z), i.e. the degree to which feedback should be attributed to the intrinsic reward distribution rather than to the latent agent
            if i_c==2:
                if i_r==0:
                    prob_attribution=1 #Benevolent agent (negative feedback)
                elif i_r==1:
                    prob_attribution = (Q[i_a]*(1-prob_intervention)) / (Q[i_a]*(1-prob_intervention)+prob_intervention) #Benevolent agent (positive feedback)
            elif i_c==1:
                if i_r==0:
                    prob_attribution = ((1-Q[i_a])*(1-prob_intervention))/((1-Q[i_a])*(1-prob_intervention)+prob_intervention) #Adversarial agent (negative feedback)
                elif i_r==1:
                    prob_attribution=1 #Adversarial agent (positive feedback)
            elif i_c==3:
                if i_r==0:
                    prob_attribution=((1-Q[i_a])*(1-prob_intervention))/((1-Q[i_a])*(1-prob_intervention)+prob_intervention/2) #Random agent (negative feedback)   
                elif i_r==1:
                    prob_attribution=(Q[i_a]*(1-prob_intervention))/(Q[i_a]*(1-prob_intervention)+prob_intervention/2) #Random agent (positive feedback)
                
            #sum of past beliefs about latent agent non-intervention
            prob_attribution_cumulative[i_a] = prob_attribution_cumulative[i_a] + prob_attribution
            learning_rate = prob_attribution/(prob_attribution_cumulative[i_a] + alpha_lr_prior + beta_lr_prior)
            
            
            # Update the Q values for the next trial
            Q[i_a] = Q[i_a] + learning_rate * (i_r - Q[i_a])
                                   
            
            
        # Return the negative log likelihood of all observed actions
        return -np.sum(log_prob_actions[1:])
    
    
    x0 = [4, 1] # values close to the true mean values reported in Dorfman et al. 2018
    result = sp.optimize.minimize(llik_td, x0, args=(actions, rewards, conditions, blocks), method='BFGS')
    # print(result)
    # print('')
    # print(f'MLE: alpha = {result.x[0]} (prior = {x0[0]})')
    # print(f'MLE: beta = {result.x[1]} (prior = {x0[1]})')
    
    x_model=result.x
    
    dt_Q=pd.DataFrame()
    dt_learning_rate=pd.DataFrame()
    dt_p_attribution=pd.DataFrame()
    
    for i_t, (i_a, i_r, i_c, i_b) in enumerate(zip(actions,rewards,conditions,blocks)):
        
        # redefine parameters to intial values at first trial of each block
        #### still having some problems with this if statements should really only happen for 3 times once per block
        if i_t==0:
            var_maybeChoice=np.zeros((1,2))
            Q = np.array([.5, .5])
            prob_attribution_cumulative=np.array([alpha_lr_prior+beta_lr_prior,alpha_lr_prior+beta_lr_prior],dtype="float64")
        elif i_t>0 and i_b!=blocks.iloc[i_t-1]:
            var_maybeChoice=np.zeros((1,2))
            Q = np.array([.5, .5])
            prob_attribution_cumulative=np.array([alpha_lr_prior+beta_lr_prior,alpha_lr_prior+beta_lr_prior],dtype="float64")
    
        #condition-specific definition of P(z|r,z), i.e. the degree to which feedback should be attributed to the intrinsic reward distribution rather than to the latent agent
        if i_c==2:
            if i_r==0:
                prob_attribution=1 #Benevolent agent (negative feedback)
            elif i_r==1:
                prob_attribution = (Q[i_a]*(1-prob_intervention)) / (Q[i_a]*(1-prob_intervention)+prob_intervention) #Benevolent agent (positive feedback)
        elif i_c==1:
            if i_r==0:
                prob_attribution = ((1-Q[i_a])*(1-prob_intervention))/((1-Q[i_a])*(1-prob_intervention)+prob_intervention) #Adversarial agent (negative feedback)
            elif i_r==1:
                prob_attribution=1 #Adversarial agent (positive feedback)
        elif i_c==3:
            if i_r==0:
                prob_attribution=((1-Q[i_a])*(1-prob_intervention))/((1-Q[i_a])*(1-prob_intervention)+prob_intervention/2) #Random agent (negative feedback)   
            elif i_r==1:
                prob_attribution=(Q[i_a]*(1-prob_intervention))/(Q[i_a]*(1-prob_intervention)+prob_intervention/2) #Random agent (positive feedback)
            
        #sum of past beliefs about latent agent non-intervention
        prob_attribution_cumulative[i_a] = prob_attribution_cumulative[i_a] + prob_attribution
        learning_rate = prob_attribution/(prob_attribution_cumulative[i_a] + alpha_lr_prior + beta_lr_prior)
        
        dt_learning_rate=dt_learning_rate.append(pd.concat([pd.Series(learning_rate),pd.Series(i_c),pd.Series(i_r),pd.Series(dtf_all.subject.unique()[isubj])],axis=1),ignore_index=True)
        dt_p_attribution=dt_p_attribution.append(pd.concat([pd.Series(prob_attribution),pd.Series(i_c),pd.Series(i_r),pd.Series(dtf_all.subject.unique()[isubj])],axis=1),ignore_index=True)
        
        # Update the Q values for the next trial
        Q[i_a] = Q[i_a] + learning_rate * (i_r - Q[i_a])
    
    dt_Grand_learning_rate=dt_Grand_learning_rate.append(dt_learning_rate)
    dt_Grand_p_attribution=dt_Grand_p_attribution.append(dt_p_attribution)
                        
dt_Grand_learning_rate=dt_Grand_learning_rate.rename(columns={0:"learning_rate",1:"condition",2:"feedback",3:"subject"})
dt_Grand_p_attribution=dt_Grand_p_attribution.rename(columns={0:"Intervention_Probability",1:"condition",2:"feedback",3:"subject"})

dt_plt_lr=dt_Grand_learning_rate.groupby(by=['condition','feedback','subject']).mean().reset_index()#[['condition','feedback','learning_rate']] \
dt_plt_interven_P=dt_Grand_p_attribution.groupby(by=['condition','feedback','subject']).mean().reset_index()
dt_plt_interven_P['Intervention_Probability']=1-dt_plt_interven_P['Intervention_Probability'] # subject belief of the intervention is 1 - probablity of non-intervention

#for better legend labels 
# dt_plt_lr['Feedback']=dt_plt_lr['feedback'].transform(lambda x: 'Positive' if x == 0 else 'Negative')

diff1=dt_plt_lr[dt_plt_lr['condition']==1]['learning_rate']-dt_plt_lr[dt_plt_lr['condition']==3]['learning_rate'].reset_index(drop=True)

diff2=dt_plt_lr[dt_plt_lr['condition']==2]['learning_rate']-dt_plt_lr[dt_plt_lr['condition']==3].set_index(dt_plt_lr[dt_plt_lr['condition']==2].index)['learning_rate']


dt_plt_lr_2=pd.DataFrame({'Relative_learning_rate': diff1.append(diff2)}).join(dt_plt_lr[['condition','feedback']])

### plotting figure 2 and 3

plt.figure(figsize=[10,4])
# plt.ylim(0,1)
# plt.yticks([0,0.5,1])

g1=sns.barplot(x="condition",y="learning_rate",hue="feedback",palette={"dimgray","darkgray"},hue_order=[1,0],data=dt_plt_lr,ci=68,capsize=.02)
g1.set_xticklabels(['Adversarial','Benvolent','Neutral'])
legend_elements = [Line2D([0], [0], color='dimgray', lw=4, label='Positive'), Line2D([0], [0], color='darkgray', lw=4, label='Negative')]
g1.legend(handles=legend_elements, title='feedback')
plt.ylim(0,1)
plt.yticks([0,0.5,1])

g1.figure.savefig(datadir+'../../dorfman2019/code/Fig3a.jpg')

plt.figure(figsize=[7,4])
g2=sns.barplot(x="condition",y="Relative_learning_rate",hue="feedback",palette={"dimgray","darkgray"},hue_order=[1,0],data=dt_plt_lr_2,ci=68,capsize=.02)
g2.set_xticklabels(['Adversarial','Benvolent'])
legend_elements = [Line2D([0], [0], color='dimgray', lw=4, label='Positive'), Line2D([0], [0], color='darkgray', lw=4, label='Negative')]
g2.legend(handles=legend_elements, title='feedback')
plt.ylim(-0.15,0.15)
plt.yticks([-0.1,0,0.1])

g2.figure.savefig(datadir+'../../dorfman2019/code/Fig3b.jpg')


plt.figure(figsize=[10,4])
g3=sns.barplot(x="condition",y="Intervention_Probability",hue="feedback",palette={"dimgray","darkgray"}, hue_order=[1,0],data=dt_plt_interven_P,ci=68,capsize=.02)
g3.set_xticklabels(['Adversarial','Benvolent','Neutral'])
plt.ylim(0,1)
plt.yticks([0,0.5,1])

legend_elements = [Line2D([0], [0], color='dimgray', lw=4, label='Positive'), Line2D([0], [0], color='darkgray', lw=4, label='Negative')]
g3.legend(handles=legend_elements, title='feedback')

g3.figure.savefig(datadir+'../../dorfman2019/code/Fig2b.jpg')


### reproducing stats of 2 way anova using difference score

model = ols('Relative_learning_rate ~ C(condition) + C(feedback) + C(condition):C(feedback)', data=dt_plt_lr_2).fit()
sm.stats.anova_lm(model, typ=2)






