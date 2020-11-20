#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:08:19 2020

@author: huanwang
"""

import pandas as pd
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from matplotlib.lines import Line2D

datadir='/Users/huanwang/GoogleDrive/Stanford/1st_yr/251/valence-control/data/'

dt=pd.read_csv(datadir+'exp1_data.csv')



dt_lg_stat1=dt.groupby(['subject','feedback']).mean()['latent_guess'].unstack()

dt_lg_stat2=dt.groupby(['subject','condition','feedback']).mean()['latent_guess'].unstack().unstack()

behav_stat1=stats.ttest_rel(dt_lg_stat1[0],dt_lg_stat1[1])  #different from reported stats "t(71) = 16.82, p < .0001"
behav_stat1_d=(dt_lg_stat1.mean().diff()[1])/(np.sqrt(sum(dt_lg_stat1.std().apply(np.sqrt))/2))

#behav_stat1_ci=t.interval(0.95,len(dt_lg_stat1)-1,loc=behav_stat1_d, scale=dt_lg_stat1.sem())


behav_stat2=[stats.ttest_rel(dt_lg_stat2[0][1],dt_lg_stat2[1][1]),\
             stats.ttest_rel(dt_lg_stat2[1][2],dt_lg_stat2[0][2]),\
             stats.ttest_rel(dt_lg_stat2[0][3],dt_lg_stat2[1][3])]

dt_fig2a=dt_lg_stat2.melt(value_name="latent_guess",ignore_index=False)

plt.figure(figsize=[10,4])
plt.ylim(0,1)
plt.yticks([0,0.5,1])
ax=sns.barplot(x="condition",y="latent_guess",hue="feedback",palette={"darkgray","dimgray"},hue_order=[1,0],data=dt_fig2a,ci=68,capsize=.02)
legend_elements = [Line2D([0], [0], color='darkgray', lw=4, label='Positive'), Line2D([0], [0], color='dimgray', lw=4, label='Negative')]
ax.legend(handles=legend_elements, title='feedback')
ax.set_xticklabels(['Adversarial','Benvolent','Neutral'])

ax.figure.savefig(datadir+'../../dorfman2019/code/Fig2a.jpg')



