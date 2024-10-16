# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:32:50 2024

@author: ZEL45
"""

# plot phase diagram: search efficiency versus Rv and rho

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
Rv_series = [0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05]
NumCases = 50

agg_effi = np.zeros([np.size(rho_series), np.size(Rv_series)])
std_effi = np.zeros([np.size(rho_series), np.size(Rv_series)])
agg_fe = np.zeros_like(agg_effi)
std_fe = np.zeros_like(agg_effi)
agg_ft = np.zeros_like(agg_effi)
std_ft = np.zeros_like(agg_effi)
tw_len = np.zeros_like(agg_effi) # length of targeted walk
std_tw_len = np.zeros_like(agg_effi)

for m in range(np.size(Rv_series)):
    Rv = Rv_series[m]
    for n in range(np.size(rho_series)):
        rho = rho_series[n]
        file = 'Rv'+str(Rv)+'rho='+str(rho)+'_simulation20.h5'
        # Open the HDF5 file in read mode
        with h5py.File(file, 'r') as hdf:
            # List all groups
            existing_groups = list(hdf.keys())
            # print(f"Existing groups: {existing_groups}")          
            # Count the number of groups
            num_groups = len(existing_groups)
            print(f"Number of groups: {num_groups}")
            fe = np.zeros(NumCases) #　ratio of exploiter mu = 3　
            ft = np.zeros(NumCases) # ratio of agents performing targeted walk
            effi = np.zeros(NumCases)
            agg_tw = []
            for i in range(NumCases):
                groupname = 'case_' + str(i)
                # Get a group or dataset
                group = hdf.get(groupname)
                numtar = group.get('foods')
                time = group.get('ticks')
                
                subgroup = group["subgroup_mu"]
                # Access and read a dataset within the subgroup
                mugroup = subgroup['mu'][:]
                
                twgroup = group["tw_step_lengths"]
                # Access and read a dataset within the subgroup
                tw = twgroup['step'][:]
                
                for k in range(np.size(tw,1)):
                    arrary = tw[:,k]
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
                
                fe[i] = np.count_nonzero(mugroup == 3)/np.size(mugroup)
                ft[i] = np.count_nonzero(tw)/np.size(tw)
                effi[i] = numtar[-1]/time[-1]
            
            agg_effi[n,m] = np.mean(effi)
            std_effi[n,m] = 1.96 * np.std(agg_effi)/np.sqrt(NumCases)
            agg_fe[n,m] = np.mean(fe)
            std_fe[n,m] = 1.96 * np.std(fe)/np.sqrt(NumCases)
            agg_ft[n,m] = np.mean(ft)
            std_ft[n,m] = 1.96 * np.std(ft)/np.sqrt(NumCases)
            tw_len[n,m] = np.mean(agg_tw_len)
            std_tw_len[n,m] = 1.96 * np.std(agg_tw_len)/np.sqrt(np.size(agg_tw_len))

#%% plot phase diagram
X,Y = np.meshgrid(Rv_series, rho_series)
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, agg_effi, 20, cmap='coolwarm')  # Use colormap 'viridis' or others like 'plasma', 'inferno', etc.
plt.colorbar()
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$\rho$', fontsize=20)
plt.savefig('Effi-R-rho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, agg_fe, 20, cmap='coolwarm')  # Use colormap 'viridis' or others like 'plasma', 'inferno', etc.
plt.colorbar()
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$\rho$', fontsize=20)
plt.savefig('fe-R-rho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, agg_ft, 20, cmap='coolwarm')  # Use colormap 'viridis' or others like 'plasma', 'inferno', etc.
plt.colorbar()
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$\rho$', fontsize=20)
plt.savefig('ft-R-rho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, tw_len, 20, cmap='coolwarm')  # Use colormap 'viridis' or others like 'plasma', 'inferno', etc.
plt.colorbar()
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$\rho$', fontsize=20)
plt.savefig('TwLen-R-rho.png', dpi=600, bbox_inches='tight')
plt.show()

#%% plot lines
plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(rho_series, agg_effi[:,p], yerr=std_effi[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='lower right', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(Rv_series, agg_effi[p,:], yerr=std_effi[p,:], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label=r'$\rho=$'+str(rho_series[p]))
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper right', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-varingrho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(rho_series, agg_fe[:,p], yerr=std_fe[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$f_{\mu=3}$', fontsize=20)
plt.legend(loc='lower right', ncol=2, fontsize=12, frameon=True)
plt.savefig('fe-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(Rv_series, agg_fe[p,:], yerr=std_fe[p,:], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label=r'$\rho=$'+str(rho_series[p]))
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$f_{\mu=3}$', fontsize=20)
plt.legend(loc='lower right', ncol=2, fontsize=12, frameon=True)
plt.savefig('fe-varingrho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(rho_series, agg_ft[:,p], yerr=std_ft[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$f_t$', fontsize=20)
plt.legend(loc='lower right', ncol=2, fontsize=12,  frameon=True)
plt.savefig('ft-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(Rv_series, agg_ft[p,:], yerr=std_ft[p,:], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label=r'$\rho=$'+str(rho_series[p]))
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$f_t$', fontsize=20)
plt.legend(loc='upper right', ncol=2, fontsize=12, frameon=True)
plt.savefig('ft-varingrho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(rho_series, tw_len[:,p], yerr=std_tw_len[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
plt.xlabel(r'$\rho$', fontsize=20)
plt.ylabel(r'$\left\langle l_t \right\rangle$', fontsize=20)
plt.legend(loc='lower right', ncol=2, fontsize=12, frameon=True)
plt.savefig('twlen-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(Rv_series, tw_len[p,:], yerr=std_tw_len[p,:], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label=r'$\rho=$'+str(rho_series[p]))
plt.xlabel(r'$R$', fontsize=20)
plt.ylabel(r'$\left\langle l_t \right\rangle$', fontsize=20)
plt.legend(loc='upper right', ncol=2, fontsize=12, frameon=True)
plt.savefig('twlen-varingrho.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(agg_fe[:,p], agg_effi[:,p], xerr=std_fe[:,p], yerr=std_effi[:,p], fmt='o', capsize=5, capthick=2, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
    # plt.errorbar(rho_series, tw_len[:,p], yerr=std_tw_len[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='Rv='+str(Rv_series[p]))
plt.xlabel(r'$f_{\mu=3}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-fe-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(agg_ft[:,p], agg_effi[:,p], xerr=std_ft[:,p], yerr=std_effi[:,p], fmt='o', capsize=5, capthick=2, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
    # plt.errorbar(rho_series, tw_len[:,p], yerr=std_tw_len[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='Rv='+str(Rv_series[p]))
plt.xlabel(r'$f_t$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-ft-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(agg_fe[:,p], agg_effi[:,p], xerr=std_fe[:,p], yerr=std_effi[:,p], fmt='o', capsize=5, capthick=2, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
    # plt.errorbar(rho_series, tw_len[:,p], yerr=std_tw_len[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='Rv='+str(Rv_series[p]))
plt.xlabel(r'$f_{\mu=3}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-fe-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(agg_ft[:,p], agg_effi[:,p], xerr=std_ft[:,p], yerr=std_effi[:,p], fmt='o', capsize=5, capthick=2, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
    # plt.errorbar(rho_series, tw_len[:,p], yerr=std_tw_len[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='Rv='+str(Rv_series[p]))
plt.xlabel(r'$f_t$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-ft-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    # Plot with customized error bars
    plt.errorbar(tw_len[:,p], agg_effi[:,p], xerr=std_tw_len[:,p], yerr=std_effi[:,p], fmt='o', capsize=5, capthick=2, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
    # plt.errorbar(rho_series, tw_len[:,p], yerr=std_tw_len[:,p], fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2, label='Rv='+str(Rv_series[p]))
plt.xlabel(r'$\left\langle l_t \right\rangle$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('twlen-ft-varingR.png', dpi=600, bbox_inches='tight')
plt.show()
#%% Lines plot without errorbar
plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    plt.plot(tw_len[:,p], agg_effi[:,p], label='R='+str(Rv_series[p]))
    plt.fill_between(tw_len[:,p], agg_effi[:,p] - std_effi[:,p], agg_effi[:,p] + std_effi[:,p], alpha=0.2)
plt.xlabel(r'$\left\langle l_t \right\rangle$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('twlen-ft-varingR.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    plt.plot(agg_fe[:,p], agg_effi[:,p], label='R='+str(Rv_series[p]))
    plt.fill_between(agg_fe[:,p], agg_effi[:,p] - std_effi[:,p], agg_effi[:,p] + std_effi[:,p], alpha=0.2)
plt.xlabel(r'$f_{\mu=3}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-fe-varingR.png', dpi=600, bbox_inches='tight')
plt.show()
#%% normalized
plt.figure(figsize=(10, 6))
for p in range(np.size(Rv_series)):
    facx = agg_fe[1,p]
    facy = agg_effi[1,p]
    plt.errorbar(agg_fe[:,p]/facx, agg_effi[:,p]/facy, xerr=std_fe[:,p], yerr=std_effi[:,p], fmt='o', capsize=5, capthick=2, elinewidth=2, markeredgewidth=2, label='R='+str(Rv_series[p]))
plt.xlabel(r'$f_{\mu=3}$', fontsize=20)
plt.ylabel(r'$\eta$', fontsize=20)
plt.legend(loc='upper left', ncol=2, fontsize=12, frameon=True)
plt.savefig('eta-fe-varingR.png', dpi=600, bbox_inches='tight')
plt.show()