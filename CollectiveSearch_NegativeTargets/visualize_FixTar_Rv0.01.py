# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:18:49 2024

@author: Starr
"""

from plotfunc import timedata,draw_video
from IPython.display import display
import ipywidgets as widgets
import numpy as np
import h5py

# rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7] 
rho_series = [0.01, 0.03] 
Rv_series = [0.01]

targets = 2500
beta = 2
agents = 10
mu = 1.1
runs = 1
file = "A10P20NP20.h5"
tick_speed = 1
alpha = 1e-5
Rv = 0.01
rho = 0.2
save = None

NumCases = 100
TarFixFile = '20P_Negative_20Patches.h5' #targets = 10000　beta = 2

#%%
for m in range(np.size(Rv_series)):
    Rv = Rv_series[m]
    for n in range(np.size(rho_series)):
        rho = rho_series[n]
        filename = 'Rv' + str(Rv) + 'rho=' + str(rho) + '_' + file
        with h5py.File(TarFixFile, 'r') as hdf:
            # lList all groups
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
                nloc_x = group.get('ntx') # get the x-coordinate
                nloc_y = group.get('nty') # get the y-coordinate
                loc_x = loc_x[:]
                loc_y = loc_y[:]
                nloc_x = nloc_x[:]
                nloc_y = nloc_y[:]
                target_location = np.column_stack((loc_x, loc_y))
                Negative_targets = np.column_stack((nloc_x, nloc_y))
                CaseCount = i
                timedata(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, filename, CaseCount, target_location, Negative_targets, save) # one run for one target distribution
#%%
NumCases = 1
tick_speed = 5
rho = 0.2
runs = 1
file = 'A10P5T2500_2.mp4'
for m in range(np.size(Rv_series)):
    Rv = Rv_series[m]
    # for n in range(np.size(rho_series)):
    #     rho = rho_series[n]
    videofilename = 'Rv' + str(Rv) + 'rho=' + str(rho) + '_' + file
    with h5py.File(TarFixFile, 'r') as hdf:
        # List all groups
        existing_groups = list(hdf.keys()) 
        # Count the number of groups
        num_groups = len(existing_groups)
        print(f"Number of groups: {num_groups}")
        for i in [2]: #range(NumCases):
            groupname = 'case_' + str(i)
            # Get a group or dataset
            group = hdf.get(groupname)
            loc_x = group.get('tx') # get the x-coordinate
            loc_y = group.get('ty') # get the y-coordinate
            nloc_x = group.get('ntx') # get the x-coordinate
            nloc_y = group.get('nty') # get the y-coordinate
            loc_x = loc_x[:]
            loc_y = loc_y[:]
            nloc_x = nloc_x[:]
            nloc_y = nloc_y[:]
            target_location = np.column_stack((loc_x, loc_y))
            Negative_targets = np.column_stack((nloc_x, nloc_y))
            draw_video(targets, beta, agents, mu, alpha, rho, Rv, tick_speed, runs, videofilename, target_location, Negative_targets)