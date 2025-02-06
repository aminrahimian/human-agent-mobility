# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:08:57 2024

@author: ZEL45
"""

# Fixed number of patches with different number of agents
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# rho_series = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7] 
# Rv = 0.05
rho_series = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7] 
Rv = 0.01
NumCases = 100
base_dir = os.path.dirname(os.path.abspath('PhaseDiagram.py'))

agg_effi = np.zeros(np.size(rho_series))
std_effi = np.zeros_like(agg_effi)
agg_fe = np.zeros_like(agg_effi)
std_fe = np.zeros_like(agg_effi)
agg_ft = np.zeros_like(agg_effi)
std_ft = np.zeros_like(agg_effi)
tw_len = np.zeros_like(agg_effi) # length of targeted walk
std_tw_len = np.zeros_like(agg_effi)
agg_neg = np.zeros([np.size(rho_series), NumCases]) # number of collected negative targets
agg_nor = np.zeros_like(agg_neg) # number of collected normal targets
agg_time = np.zeros_like(agg_neg) # search time

for k in range(np.size(rho_series)):
    rho = rho_series[k]
    file = 'Rv'+str(Rv)+'rho='+str(rho)+'_A10P20NP20.h5'
    id_range = range(NumCases)
    with h5py.File(file, 'r') as hdf:
        existing_groups = list(hdf.keys())
        # print(f"Existing groups: {existing_groups}")          
        # Count the number of groups
        num_groups = len(existing_groups)
        print(f"Number of groups: {num_groups}")
        fe = np.zeros(NumCases) #　ratio of exploiter mu = 3　
        ft = np.zeros(NumCases) # ratio of agents performing targeted walk
        effi = np.zeros(NumCases)
        agg_tw = []
        numNegCol = np.zeros(NumCases)
        for i in id_range:
            groupname = 'case_' + str(i)
            # Get a group or dataset
            group = hdf.get(groupname)
            numtar = group.get('foods')
            time = group.get('ticks')
            numNegtive = group.get('NumNegTar')
            
            subgroup = group["subgroup_mu"]
            # Access and read a dataset within the subgroup
            mugroup = subgroup['mu'][:]
            
            twgroup = group["tw_step_lengths"]
            # Access and read a dataset within the subgroup
            tw = twgroup['step'][:]
            
            NegRew = np.size(numNegtive,0) * 100 # negative rewards
            for kk in range(np.size(tw,1)):
                arrary = tw[:,kk]
                max_values = []
                temp = []
                for value in arrary:
                    if value != 0:
                        temp.append(value)
                    else:
                        if temp:
                            max_values.append(max(temp))
                            temp = []
                if temp:
                    max_values.append(max(temp))
                agg_tw.append(max_values)
            
            # Flatten the list of lists into a single list
            flattened_list = [item for sublist in agg_tw for item in sublist]                
            # Convert the flattened list to a NumPy one-dimensional array
            agg_tw_len = np.array(flattened_list)
            # Convert dataset to a NumPy array or access the data
            numtar = numtar[:]
            time = time[:]
            
            fe[i-id_range[0]] = np.count_nonzero(mugroup == 3)/np.size(mugroup)
            ft[i-id_range[0]] = np.count_nonzero(tw)/np.size(tw)
            effi[i-id_range[0]] = (numtar[-1])/time[-1]
            
            agg_neg[k, i-id_range[0]] = np.size(numNegtive,0)
            agg_nor[k, i-id_range[0]] = numtar[-1]
            agg_time[k, i-id_range[0]] = time[-1]
            
    agg_effi[k] = np.mean(effi)
    std_effi[k] = 1.96 * np.std(agg_effi)/np.sqrt(NumCases)
    agg_fe[k] = np.mean(fe)
    std_fe[k] = 1.96 * np.std(fe)/np.sqrt(NumCases)
    agg_ft[k] = np.mean(ft)
    std_ft[k] = 1.96 * np.std(ft)/np.sqrt(NumCases)
    tw_len[k] = np.mean(agg_tw_len)
    std_tw_len[k] = 1.96 * np.std(agg_tw_len)/np.sqrt(np.size(agg_tw_len))
    


#%% plot number of collected negative targets with rho
rho_series_com = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
effi_com = [0.225107,
            0.24634,
            0.338751,
            0.45288,
            0.503193,
            0.526244,
            0.518446,
            0.556711,
            0.442852,
            0.406218,
            0.367285,
            ]
stdeffi_com = [0.0712857,
            0.0709167,
            0.0705434,
            0.0702661,
            0.0700394,
            0.0698129,
            0.0695193,
            0.0692552,
            0.068748,
            0.0681712,
            0.0675449,
            ]

plt.figure(figsize=(8, 6))
x_series = [0,5,10,20]
# Sample 6 evenly spaced colors from the 'viridis' colormap
cmap = plt.get_cmap("viridis")
colors = [cmap(i) for i in np.linspace(0, 1, len(x_series))]
for i in range(len(x_series)):
    agg_effi = (agg_nor-x_series[i]*agg_neg)/agg_time
    plt.errorbar(
        rho_series, np.mean(agg_effi,1), yerr=1.96*np.std(agg_effi,1)/np.sqrt(NumCases),
        color=colors[i],         # Line color
        ecolor=colors[i],         # Error bar color
        elinewidth=2,       # Error bar line width
        capsize=5,            # Error bar cap size
        marker='o',           # Marker style
        mfc=colors[i],          # Marker face color
        mec=colors[i],          # Marker edge color
        mew=2,              # Marker edge width
        label = 'X='+str(x_series[i])
    )
    
plt.errorbar(
    rho_series_com, effi_com, yerr=stdeffi_com,
    color='red',         # Line color
    ecolor='red',         # Error bar color
    elinewidth=2,       # Error bar line width
    capsize=5,            # Error bar cap size
    marker='o',           # Marker style
    mfc='red',          # Marker face color
    mec='red',          # Marker edge color
    mew=2,              # Marker edge width
    label = 'Normal'
)
# plt.figure(figsize=(8, 6))
# plt.errorbar(rho_series, agg_effi, yerr=std_effi, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 0.8])
# plt.xlim([-0.02, 0.72])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=16, frameon=True)
plt.savefig('NumNeg-rho'+'.png', dpi=600, bbox_inches='tight')
plt.show()
# #%% plot spectrum for efficiency curves considering negative targets
# plt.figure(figsize=(8, 6))
# for i in np.arange(0,50,10):
#     plt.errorbar(
#         rho_series, agg_neg, yerr=std_neg,
#         color='red',         # Line color
#         ecolor='red',         # Error bar color
#         elinewidth=2,       # Error bar line width
#         capsize=5,            # Error bar cap size
#         marker='o',           # Marker style
#         mfc='red',          # Marker face color
#         mec='red',          # Marker edge color
#         mew=2,              # Marker edge width
#     )

# plt.xlabel(r'$\rho$', fontsize=20)
# plt.ylabel(r'$N_n$', fontsize=20)
# # plt.legend(loc='lower right', ncol=1, fontsize=16, frameon=True)
# plt.savefig('NumNeg-rho'+'.png', dpi=600, bbox_inches='tight')
# plt.show()
#%% plor eta-rho for fixed number of agents
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

rho_series_com = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
effi_com = [0.225107,
            0.24634,
            0.338751,
            0.45288,
            0.503193,
            0.526244,
            0.518446,
            0.556711,
            0.442852,
            0.406218,
            0.367285,
            ]
stdeffi_com = [0.0712857,
            0.0709167,
            0.0705434,
            0.0702661,
            0.0700394,
            0.0698129,
            0.0695193,
            0.0692552,
            0.068748,
            0.0681712,
            0.0675449,
            ]

plt.figure(figsize=(8, 6))
plt.errorbar(
    rho_series, agg_effi, yerr=std_effi,
    color='blue',         # Line color
    ecolor='blue',         # Error bar color
    elinewidth=2,       # Error bar line width
    capsize=5,            # Error bar cap size
    marker='o',           # Marker style
    mfc='blue',          # Marker face color
    mec='blue',          # Marker edge color
    mew=2,               # Marker edge width
    label='With negative patches' 
)
plt.errorbar(
    rho_series_com, effi_com, yerr=stdeffi_com,
    color='red',         # Line color
    ecolor='red',         # Error bar color
    elinewidth=2,       # Error bar line width
    capsize=5,            # Error bar cap size
    marker='o',           # Marker style
    mfc='red',          # Marker face color
    mec='red',          # Marker edge color
    mew=2,              # Marker edge width
    label = 'No negative patches'
)
# plt.figure(figsize=(8, 6))
# plt.errorbar(rho_series, agg_effi, yerr=std_effi, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 0.8])
plt.xlim([-0.02, 0.72])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=16, frameon=True)
plt.savefig('eta-rho'+'.png', dpi=600, bbox_inches='tight')
plt.show()

#%% plor ft-rho for fixed number of agents
plt.figure(figsize=(8, 6))
plt.errorbar(rho_series, agg_ft, yerr=std_ft, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 6.5])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$f_t$', fontsize=20)
plt.legend(loc='upper right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-ft'+'.png', dpi=600, bbox_inches='tight')
plt.show()

#%% plot eta-fe
plt.figure(figsize=(8, 6))
plt.scatter(agg_fe, agg_effi)
# plt.ylim([0, 6.5])
plt.xlabel(r'$f_{\mu=3}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-fmu=3'+'.png', dpi=600, bbox_inches='tight')
plt.show()

# plot eta-fT
plt.figure(figsize=(8, 6))
plt.scatter(agg_ft, agg_effi)
# plt.ylim([0, 6.5])
plt.xlabel(r'$f_t$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-ft'+'.png', dpi=600, bbox_inches='tight')
plt.show()

# plot eta-fmu=1.1
plt.figure(figsize=(8, 6))
plt.scatter(agg_ft, agg_effi)
# plt.ylim([0, 6.5])
plt.xlabel(r'$f_{\mu=1.1}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-fmu=1.1'+'.png', dpi=600, bbox_inches='tight')
plt.show()
#%% plor fe-rho for fixed number of agents
plt.figure(figsize=(8, 6))
plt.errorbar(rho_series, agg_fe, yerr=std_fe, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 6.5])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$f_{\mu=3}$', fontsize=20)
plt.legend(loc='upper right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-fmu=3'+'.png', dpi=600, bbox_inches='tight')
plt.show()

#%% plor twlen-rho for fixed number of agents
plt.figure(figsize=(8, 6))
plt.errorbar(rho_series, tw_len, yerr=std_tw_len, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 6.5])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\left\langle l_t \right\rangle$', fontsize=20)
plt.legend(loc='upper right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-Twlen'+'.png', dpi=600, bbox_inches='tight')
plt.show()
#%% comparing cases with the same number of targets in each patch
filepaths = ['5Patches', '10Patches', r'$20\mathrm{P}\times 500\mathrm{T}$', r'$10\mathrm{P}\times 500\mathrm{T}$', r'$5\mathrm{P}\times 500\mathrm{T}$']
plt.figure(figsize=(8, 6))
plt.errorbar(rho_series, agg_effi, yerr=std_effi, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 6.5])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-rho'+'.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.errorbar(rho_series, agg_ft, yerr=std_ft, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 6.5])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$f_t$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-ft'+'.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(agg_fe, agg_effi)
# plt.ylim([0, 6.5])
plt.xlabel(r'$f_{\mu=3}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-fmu=3'+'.png', dpi=600, bbox_inches='tight')
plt.show()

# plot eta-fT
plt.figure(figsize=(8, 6))
plt.scatter(agg_ft, agg_effi)
# plt.ylim([0, 6.5])
plt.xlabel(r'$f_t$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-ft'+'.png', dpi=600, bbox_inches='tight')
plt.show()

# plot eta-fmu=1.1
plt.figure(figsize=(8, 6))
plt.scatter(agg_ft, agg_effi)
# plt.ylim([0, 6.5])
plt.xlabel(r'$f_{\mu=1.1}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=12, frameon=True)
plt.savefig('eta-fmu=1.1'+'.png', dpi=600, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 6))
plt.errorbar(rho_series, tw_len, yerr=std_tw_len, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.ylim([0, 6.5])
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\left\langle l_t \right\rangle$', fontsize=20)
plt.legend(loc='lower right', ncol=1, fontsize=12, frameon=True)