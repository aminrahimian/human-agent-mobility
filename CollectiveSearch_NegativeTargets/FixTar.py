# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:13:09 2024

@author: ZEL45
"""

# generate fixed target distribution
import numpy as np
import itertools
from MainFixTar_VarPatch import target_structure
import h5py

targets = 10000
beta = 2
agents = 10
mu = 1.1
runs = 20
alpha = 1e-6
Rvs = [0.1]
rhos = [0.2]
save = True
num_patches = 20

radius_interaction = [1]

trees = [targets]
betas = [beta]
agents = [agents]
mu_range = [mu]
alpha_range = [float(alpha)]
paramlist = list(itertools.product(trees, agents, betas, radius_interaction, mu_range, alpha_range, rhos, Rvs))
print(paramlist[0])
input_paras = paramlist[0]
input_paras = (2500,) + input_paras[1:]

NumTar = 100
hdf5_filename = '10P_Negative_5Patches.h5'
for i in range(NumTar):
    [tarx, tary] = target_structure(paramlist[0], num_patches)
    [Ntarx, Ntary] = target_structure(input_paras, 5)
    case_id = i
    with h5py.File(hdf5_filename, 'a') as hdf5_file:
        # List all groups
        print("Keys: %s" % hdf5_file.keys())
        # Create a group for each case using the case_id
        group_name = f"case_{case_id}"
        case_group = hdf5_file.create_group(group_name)
        
        # Save ticks and efficiencies as datasets
        case_group.create_dataset('tx', data=np.array(tarx), compression="gzip", compression_opts=9)
        case_group.create_dataset('ty', data=np.array(tary), compression="gzip", compression_opts=9)
        case_group.create_dataset('ntx', data=np.array(Ntarx), compression="gzip", compression_opts=9)
        case_group.create_dataset('nty', data=np.array(Ntary), compression="gzip", compression_opts=9)
