#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:11:37 2020

@author: huanwang
"""

#https://pymc-devs.github.io/pymc/modelfitting.html

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


basic_model = pm.Model()
prob_intervention=0.3

data_file = '/Users/huanwang/GoogleDrive/Stanford/1st_yr/251/valence-control/data/exp1_data.csv'

dtf=pd.read_csv(data_file)
prob_attribution=pd.DataFrame(np.repeat(0,len(dtf)))
with basic_model:
    
    # empirical priors based on previous research (Gershman, 2016)
    beta=pm.Gamma('beta',alpha=4.82,beta=0.88)
    rho=pm.Normal('rho',mu=0.15,sigma=1.42)
    # prior belief of true reward probability from participants
    theta=pm.Beta('theta',alpha=1,beta=1)
    
    #condition-specific definition of P(z|r,z), i.e. the degree to which feedback should be attributed to the intrinsic reward distribution rather than to the latent agent
    
    if dtf['condition']==2:
        if dtf['feedback']==0:
            prob_attribution[(dtf['condition']==2) & (dtf['feedback']==0)]=1 #Benevolent agent (negative feedback)
        elif dtf['feedback']==1:
            prob_attribution[(dtf['condition']==2) & (dtf['feedback']==1)]=(theta*(1-prob_intervention))/(theta*(1-prob_intervention)+prob_intervention) #Benevolent agent (positive feedback)
    elif dtf['condition']==1:
        if dtf['feedback']==0:
            prob_attribution[(dtf['condition']==1) & (dtf['feedback']==0)]=((1-theta)*(1-prob_intervention))/((1-theta)*(1-prob_intervention)+prob_intervention) #Adversarial agent (negative feedback)
        elif dtf['feedback']==1:
            prob_attribution[(dtf['condition']==1) & (dtf['feedback']==1)]=1 #Adversarial agent (positive feedback)
    elif dtf['condition']==3:
        if dtf['feedback']==0:
            prob_attribution[(dtf['condition']==3) & (dtf['feedback']==0)]=((1-theta)*(1-prob_intervention))/((1-theta)*(1-prob_intervention)+prob_intervention/2) #Random agent (negative feedback)   
        elif dtf['feedback']==1:
            prob_attribution[(dtf['condition']==3) & (dtf['feedback']==1)]=(theta*(1-prob_intervention))/(theta*(1-prob_intervention)+prob_intervention/2) #Random agent (positive feedback)


    learning_rate=prob_attribution/(prob_attribution_cumulative+1+1)
    
    Yobs=pm.Bernoulli('Yobs',p=,observed=Y)









