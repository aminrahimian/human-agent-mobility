# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:39:38 2024

@author: ZEL45
"""

from plotfunc import draw_novideo, draw_video, timedata, draw_video_timedata
from IPython.display import display
import ipywidgets as widgets
import numpy as np
import h5py

rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7] 
Rv_series = [0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05]

targets = 10000
beta = 2
agents = 10
mu = 1.1
runs = 1
file = "Fix50.h5"
tick_speed = 1
alpha = 1e-5
Rv = 0.01
rho = 0.2
save = None

NumCases = 30
TarFixFile = 'TarFix.h5' #targets = 10000　beta = 2

for m in range(np.size(Rv_series)):
    Rv = Rv_series[m]
    for n in range(np.size(rho_series)):
        rho = rho_series[n]
        filename = 'Rv' + str(Rv) + 'rho=' + str(rho) + '_' + file
        with h5py.File(TarFixFile, 'r') as hdf:
            # List all groups
            existing_groups = list(hdf.keys())
            # Count the number of groups
            num_groups = len(existing_groups)
            print(f"Number of groups: {num_groups}")
            for i in range(NumCases):
                groupname = 'case_' + str(i)
                # Get a group or dataset
                group = hdf.get(groupname)
                loc_x = group.get('tx') # get the x-coordinate
                loc_y = group.get('ty') # get the y-coordinate
                loc_x = loc_x[:]
                loc_y = loc_y[:]
                target_location = np.column_stack((loc_x, loc_y))
                CaseCount = i
                timedata(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, filename, CaseCount, save) # one run for one target distribution
                