# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:32:06 2024

@author: ZEL45
"""
import h5py
import matplotlib.pyplot as plt

# test target distribution with specified number of patches
file = '20P_Negative_20Patches.h5'
with h5py.File(file, 'r') as hdf:
    groupname = f"case_{2}" 
    group = hdf.get(groupname)
    loc_x = group.get('tx') # get the x-coordinate
    loc_y = group.get('ty') # get the y-coordinate
    loc_x = loc_x[:]
    loc_y = loc_y[:]
    
    nloc_x = group.get('ntx') # get the x-coordinate
    nloc_y = group.get('nty') # get the y-coordinate
    nloc_x = nloc_x[:]
    nloc_y = nloc_y[:]
    
    plt.scatter(loc_x, loc_y, c = 'green', s=6)
    plt.scatter(nloc_x, nloc_y, c = 'red', s=6)