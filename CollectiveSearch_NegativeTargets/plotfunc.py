#!/usr/bin/env python
# coding: utf-8

# In[15]:


from __future__ import division, print_function

import random
import math
import os
import random
import shutil
import numpy as np
from scipy import spatial as st 
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
from periodic_kdtree import PeriodicCKDTree
from sklearn.cluster import DBSCAN
from scipy import stats as sts
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pylab as pl

import sys
import copy
import itertools
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pylab as pl
from IPython import display
from IPython.display import display, clear_output
# from ipynb.fs.full.main_module import main
from ModMainFix import main_timeana, main_video
# Set the interactive backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

# In[1]:


folder = "data"
radius_interaction = [1]


# In[2]:


def f(targets, beta, agents, mu, alpha, tick_speed, runs, save):
    return

def create_widgets() :
    w = interactive(f, targets = widgets.IntSlider(min=0, max=10000, step=1000, value=1000), beta = widgets.FloatSlider(min=1.1, max=3.5, step=0.1, value=3), agents = widgets.IntSlider(min=1, max=100, step=1, value=10), mu = widgets.FloatSlider(min=1.1, max=3.5, step=0.1, value=3), alpha = widgets.Text('0.001'), tick_speed = widgets.IntSlider(min=1, max=100, step=1, value=10),  runs = widgets.Text('10'), save=widgets.Checkbox(value=False, description='Save output?')  )
    return w

def timedata(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, hdf5filename, CaseCount, target_location, Negative_targets, save):
    trees = [targets]
    betas = [beta]
    agents = [agents]
    mu_range = [mu]
    alpha_range = [float(alpha)]
    rhos = [rho]
    Rvs = [Rv]
    
    var_list = [trees[0], agents[0], betas[0], mu_range[0], alpha_range[0], rhos[0], Rvs[0]]
    string_ints = [str(int) for int in var_list]
    filename = "_".join(string_ints)

    if save:
        # if os.path.exists(folder) and os.path.isdir(folder):
        #     shutil.rmtree(folder)
        # os.mkdir(folder)
        f2 = open(folder + "/" + filename + ".txt", 'a')

    z = 0
    while z < runs:
        paramlist = list(itertools.product(trees, agents, betas, radius_interaction, mu_range, alpha_range, rhos, Rvs))
        print(paramlist[0])
        caseid = z + CaseCount # number of existed groups
        main_timeana(paramlist[0], tick_speed, hdf5filename, caseid, target_location, Negative_targets)
        z += 1
        
def draw_video(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, videofilename, target_locations, Negative_targets):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots(1, 1)

    ax.set_facecolor('white')

    trees = [targets]
    betas = [beta]
    agents = [agents]
    mu_range = [mu]
    alpha_range = [float(alpha)]
    rhos = [rho]
    Rvs = [Rv]

    z = 0
    while z < runs:
        paramlist = list(itertools.product(trees, agents, betas, radius_interaction, mu_range, alpha_range, rhos, Rvs))
        print(paramlist[0])
        
        result = main_video(paramlist[0], fig, ax, tick_speed, videofilename, target_locations, Negative_targets)
        z += 1


    plt.show()