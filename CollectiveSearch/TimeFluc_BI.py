# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:08:02 2024

@author: ZEL45
"""

# check variability of efficiency with time
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

#%%
rho_series = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1]
Rv_series = [0.005,0.01,0.015,0.02]
NumCases = 20

Rv = 0.01
rho = 0.5

file = 'Rv'+str(Rv)+'rho='+str(rho)+'_simulation20.h5'
# Open the HDF5 file in read mode
with h5py.File(file, 'r') as hdf:
    # List all groups
    print("Keys: %s" % hdf.keys())
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
        
        numtar = numtar[:]
        time = time[:]
        
        subgroup = group["subgroup_mu"]
        # Access and read a dataset within the subgroup
        mugroup = subgroup['mu'][:]
        fe_inst = np.zeros(np.size(mugroup,0))
        for j in range(np.size(mugroup,0)):
            mu_inst = mugroup[j,:]
            fe_inst[j] = np.count_nonzero(mu_inst == 3)/np.size(mu_inst)
        
        numtar_diff = numtar[1:np.size(numtar)] - numtar[0:np.size(numtar)-1]
        time_diff = time[1:np.size(time)] - time[0:np.size(time)-1]
        eta_ins = numtar_diff/time_diff
        eta_ins_add = np.insert(eta_ins, 0, 0)

        # Create a DataFrame
        df = pd.DataFrame({'Time': time, 'Values': eta_ins_add})
        plt.figure(figsize=(10, 6))
        # Plot original values
        plt.plot(df['Time'], df['Values'], label='Instantaneous efficiency', color='blue', alpha=0.5)
        # Add labels and title
        plt.xlabel('Time', fontsize=20)
        plt.ylabel(r'$\eta_i$', fontsize=20)
        plt.ylim([0, 10])
        plt.savefig('inst_effi_case-'+ str(i) +'-Rv001Rho05.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        # Plot original values
        plt.plot(time, fe_inst, label='Instantaneous fe', color='blue', alpha=0.5)
        # Add labels and title
        plt.xlabel('Time', fontsize=20)
        plt.ylabel(r'$f_{\mu=3}$', fontsize=20)
        plt.ylim([0, 1])
        plt.savefig('inst_fe_case-'+ str(i) +'-Rv001Rho05.png', dpi=600, bbox_inches='tight')
        plt.show()
        
#%% plot PDF for instantaenous efficiency, fe, ft
Rv_series = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
rho_series = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
NumCases = 50

NumBins = 11;
fe_PDF = np.zeros([NumBins, np.size(rho_series)])
fe_xrange = np.zeros_like(fe_PDF)
ft_PDF = np.zeros([NumBins, np.size(rho_series)])
ft_xrange = np.zeros_like(ft_PDF)
eta_PDF = np.zeros([NumBins, np.size(rho_series)])
eta_xrange = np.zeros_like(eta_PDF)

mean_fe = np.zeros(np.size(rho_series))
mean_ft = np.zeros(np.size(rho_series))
mean_eta = np.zeros(np.size(rho_series))
std_fe = np.zeros(np.size(rho_series))
std_ft = np.zeros(np.size(rho_series))
std_eta = np.zeros(np.size(rho_series))

for m in range(np.size(Rv_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    
    fe_list = [];
    ft_list = [];
    effi_list = [];
    file = 'Rv'+str(Rv)+'rho='+str(rho)+'_simulation20.h5'
    # Open the HDF5 file in read mode
    with h5py.File(file, 'r') as hdf:
        # List all groups
        print("Keys: %s" % hdf.keys())
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
            
            numtar = numtar[:]
            time = time[:]
            
            subgroup = group["subgroup_mu"]
            # Access and read a dataset within the subgroup
            mugroup = subgroup['mu'][:]
            twgroup = group["tw_step_lengths"]
            tw = twgroup['step'][:]
            
            ft_inst = np.zeros(np.size(tw,0))
            fe_inst = np.zeros(np.size(mugroup,0))
            for j in range(np.size(mugroup,0)):
                mu_inst = mugroup[j,:]
                fe_inst[j] = np.count_nonzero(mu_inst == 3)/np.size(mu_inst)
                tw_inst = tw[j,:]
                ft_inst[j] = np.count_nonzero(tw_inst)/np.size(tw_inst)
                
            numtar_diff = numtar[1:np.size(numtar)] - numtar[0:np.size(numtar)-1]
            time_diff = time[1:np.size(time)] - time[0:np.size(time)-1]
            eta_ins = numtar_diff/time_diff
            
            # Append the array to the list
            effi_list.extend(eta_ins) 
            fe_list.extend(fe_inst)
            ft_list.extend(ft_inst)
        
        mean_eta[m] = np.mean(effi_list)
        mean_fe[m] = np.mean(fe_list)
        mean_ft[m] = np.mean(ft_list)
        
        std_eta[m] = np.std(effi_list)
        std_fe[m] = np.std(fe_list)
        std_ft[m] = np.std(ft_list)
        
        # Calculate the kernel density estimation
        kde = stats.gaussian_kde(effi_list)
        # Create a range of x values
        # x_range = np.logspace(np.log10(min(effi_list)), np.log10(max(effi_list)), NumBins)
        x_range = np.linspace(min(effi_list), max(effi_list), NumBins) 
        # Calculate the PDF
        pdf = kde(x_range)
        # Normalize the PDF
        pdf_normalized = pdf / np.sum(pdf) /(x_range[2]-x_range[1])
        eta_PDF[:,m] = pdf_normalized
        eta_xrange[:,m] = x_range
        
        kde = stats.gaussian_kde(fe_list)
        x_range = np.linspace(min(fe_list), max(fe_list), NumBins) 
        pdf = kde(x_range)
        pdf_normalized = pdf / np.sum(pdf) /(x_range[2]-x_range[1])
        fe_PDF[:,m] = pdf_normalized
        fe_xrange[:,m] = x_range
        
        kde = stats.gaussian_kde(ft_list)
        x_range = np.linspace(min(ft_list), max(ft_list), NumBins) 
        pdf = kde(x_range)
        pdf_normalized = pdf / np.sum(pdf) /(x_range[2]-x_range[1])
        ft_PDF[:,m] = pdf_normalized
        ft_xrange[:,m] = x_range
            
plt.figure(figsize=(10, 6))
for m in range(np.size(rho_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    eta_xrange_ind = eta_xrange[:,m]
    eta_PDF_ind = eta_PDF[:,m]
    eta_xrange_ind = eta_xrange_ind[eta_PDF_ind>1e-4]
    eta_PDF_ind = eta_PDF_ind[eta_PDF_ind>1e-4]
    plt.plot(eta_xrange_ind, eta_PDF_ind, '-', linewidth=3, label=r'$\rho$='+str(rho))
plt.xlabel(r'$\eta_i$', fontsize=20)
plt.ylabel('PDF', fontsize=20)
plt.legend(loc='upper right', ncol=2, fontsize=12, frameon=True)
# plt.xscale('log')
plt.yscale('log')
plt.savefig('ins-effi-PDF.png', dpi=600, bbox_inches='tight')
plt.show()
        
plt.figure(figsize=(10, 6))
for m in range(np.size(rho_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    eta_xrange_ind = ft_xrange[:,m]
    eta_PDF_ind = ft_PDF[:,m]
    eta_xrange_ind = eta_xrange_ind[eta_PDF_ind>1e-3]
    eta_PDF_ind = eta_PDF_ind[eta_PDF_ind>1e-3]
    plt.plot(eta_xrange_ind, eta_PDF_ind, '-', linewidth=3, label=r'$\rho$='+str(rho))
plt.xlabel(r'$f_t$', fontsize=20)
plt.ylabel('PDF', fontsize=20)
plt.legend(loc='upper right', ncol=2, fontsize=12, frameon=True)
# plt.xscale('log')
plt.yscale('log')
plt.savefig('ins-ft-PDF.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for m in range(np.size(rho_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    eta_xrange_ind = fe_xrange[:,m]
    eta_PDF_ind = fe_PDF[:,m]
    eta_xrange_ind = eta_xrange_ind[eta_PDF_ind>1e-3]
    eta_PDF_ind = eta_PDF_ind[eta_PDF_ind>1e-3]
    plt.plot(eta_xrange_ind, eta_PDF_ind, '-', linewidth=3, label=r'$\rho$='+str(rho))
plt.xlabel(r'$f_e$', fontsize=20)
plt.ylabel('PDF', fontsize=20)
plt.legend(loc='lower left', ncol=2, fontsize=12, frameon=True)
# plt.xscale('log')
plt.yscale('log')
plt.savefig('ins-fe-PDF.png', dpi=600, bbox_inches='tight')
plt.show()
#%% measure burstness
Rv_series = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
rho_series = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
NumCases = 50

NumBins = 100

noevent_PDF_eta = np.zeros([NumBins, np.size(rho_series)])
noevent_xrange_eta = np.zeros_like(noevent_PDF_eta)
noevent_PDF_fe = np.zeros([NumBins, np.size(rho_series)])
noevent_xrange_fe = np.zeros_like(noevent_PDF_fe)
noevent_PDF_ft = np.zeros([NumBins, np.size(rho_series)])
noevent_xrange_ft = np.zeros_like(noevent_PDF_ft)

mean_noevent_eta = np.zeros(np.size(rho_series))
std_noevent_eta = np.zeros(np.size(rho_series))
mean_noevent_ft = np.zeros(np.size(rho_series))
std_noevent_ft  = np.zeros(np.size(rho_series))
mean_noevent_fe = np.zeros(np.size(rho_series))
std_noevent_fe  = np.zeros(np.size(rho_series))

for m in range(np.size(Rv_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    
    fe_list = [];
    ft_list = [];
    effi_list = [];
    file = 'Rv'+str(Rv)+'rho='+str(rho)+'_simulation20.h5'
    # inter-event time
    inter_time_eta = []
    inter_time_fe = []
    inter_time_ft = []
    # Open the HDF5 file in read mode
    with h5py.File(file, 'r') as hdf:
        # List all groups
        print("Keys: %s" % hdf.keys())
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
            
            numtar = numtar[:]
            time = time[:]
            
            subgroup = group["subgroup_mu"]
            # Access and read a dataset within the subgroup
            mugroup = subgroup['mu'][:]
            twgroup = group["tw_step_lengths"]
            tw = twgroup['step'][:]
            
            ft_inst = np.zeros(np.size(tw,0))
            fe_inst = np.zeros(np.size(mugroup,0))
            for j in range(np.size(mugroup,0)):
                mu_inst = mugroup[j,:]
                fe_inst[j] = np.count_nonzero(mu_inst == 3)/np.size(mu_inst)
                tw_inst = tw[j,:]
                ft_inst[j] = np.count_nonzero(tw_inst)/np.size(tw_inst)
                
            numtar_diff = numtar[1:np.size(numtar)] - numtar[0:np.size(numtar)-1]
            time_diff = time[1:np.size(time)] - time[0:np.size(time)-1]
            eta_ins = numtar_diff/time_diff
            
            # extract inter-event time
            # Find the positions where the value changes
            changes = np.diff(eta_ins)
            # A sequence of zeros starts where the value changes from non-zero to 0
            zero_starts = np.where((eta_ins[:-1] != 0) & (eta_ins[1:] == 0))[0] + 1  
            # A sequence of zeros ends where the value changes from 0 to non-zero
            zero_ends = np.where((eta_ins[:-1] == 0) & (eta_ins[1:] != 0))[0] + 1
            # Handle edge cases where the array starts or ends with 0
            if eta_ins[0] == 0:
                zero_starts = np.insert(zero_starts, 0, 0)
            if eta_ins[-1] == 0:
                zero_ends = np.append(zero_ends, len(eta_ins))
            
            # Collect the lengths of the sequences of zeros
            lengths_of_zero_frames = zero_ends - zero_starts
            inter_time_eta.extend(lengths_of_zero_frames)
            
            changes = np.diff(fe_inst)
            zero_starts = np.where((fe_inst[:-1] != 0) & (fe_inst[1:] == 0))[0] + 1  
            zero_ends = np.where((fe_inst[:-1] == 0) & (fe_inst[1:] != 0))[0] + 1
            if fe_inst[0] == 0:
                zero_starts = np.insert(zero_starts, 0, 0)
            if fe_inst[-1] == 0:
                zero_ends = np.append(zero_ends, len(eta_ins))
            
            lengths_of_zero_frames = zero_ends - zero_starts
            inter_time_fe.extend(lengths_of_zero_frames)
            
            changes = np.diff(ft_inst)
            zero_starts = np.where((ft_inst[:-1] != 0) & (ft_inst[1:] == 0))[0] + 1  
            zero_ends = np.where((ft_inst[:-1] == 0) & (ft_inst[1:] != 0))[0] + 1
            if ft_inst[0] == 0:
                zero_starts = np.insert(zero_starts, 0, 0)
            if ft_inst[-1] == 0:
                zero_ends = np.append(zero_ends, len(eta_ins))
            
            lengths_of_zero_frames = zero_ends - zero_starts
            inter_time_ft.extend(lengths_of_zero_frames)
            
            
            # Append the array to the list
            effi_list.extend(eta_ins) 
            fe_list.extend(fe_inst)
            ft_list.extend(ft_inst)
            
        
        mean_noevent_eta[m] = np.mean(inter_time_eta)
        std_noevent_eta[m] = np.std(inter_time_eta)
        mean_noevent_ft[m] = np.mean(inter_time_ft)
        std_noevent_ft[m] = np.std(inter_time_ft)
        mean_noevent_fe[m] = np.mean(inter_time_fe)
        std_noevent_fe[m] = np.std(inter_time_fe)
        
        kde = stats.gaussian_kde(inter_time_eta)
        # x_range = np.linspace(min(inter_time), max(inter_time), NumBins)
        x_range = np.logspace(np.log10(min(inter_time_eta)), np.log10(max(inter_time_eta)), NumBins)
        pdf = kde(x_range)
        pdf_normalized = pdf / np.sum(pdf) /(x_range[2]-x_range[1])
        noevent_PDF_eta[:,m] = pdf_normalized
        noevent_xrange_eta[:,m] = x_range
        
        kde = stats.gaussian_kde(inter_time_fe)
        # x_range = np.linspace(min(inter_time), max(inter_time), NumBins)
        x_range = np.logspace(np.log10(min(inter_time_fe)), np.log10(max(inter_time_fe)), NumBins)
        pdf = kde(x_range)
        pdf_normalized = pdf / np.sum(pdf) /(x_range[2]-x_range[1])
        noevent_PDF_fe[:,m] = pdf_normalized
        noevent_xrange_fe[:,m] = x_range
        
        kde = stats.gaussian_kde(inter_time_ft)
        # x_range = np.linspace(min(inter_time), max(inter_time), NumBins)
        x_range = np.logspace(np.log10(min(inter_time_ft)+1), np.log10(max(inter_time_ft)), NumBins)
        pdf = kde(x_range)
        pdf_normalized = pdf / np.sum(pdf) /(x_range[2]-x_range[1])
        noevent_PDF_ft[:,m] = pdf_normalized
        noevent_xrange_ft[:,m] = x_range
        
plt.figure(figsize=(10, 6))
for m in range(np.size(rho_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    plt.plot(noevent_xrange_eta[:,m], noevent_PDF_eta[:,m], '-', linewidth=3, label=r'$\rho$='+str(rho))
plt.xlabel(r'$t_{inter-event}$', fontsize=20)
plt.ylabel('PDF', fontsize=20)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('eta-interevent-PDF.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for m in range(np.size(rho_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    plt.plot(noevent_xrange_fe[:,m], noevent_PDF_fe[:,m], '-', linewidth=3, label=r'$\rho$='+str(rho))
plt.xlabel(r'$f_e$', fontsize=20)
plt.ylabel('PDF', fontsize=20)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('fe-interevent-PDF.png', dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for m in range(np.size(rho_series)):
    rho = rho_series[m]
    Rv = Rv_series[m]
    plt.plot(noevent_xrange_ft[:,m], noevent_PDF_ft[:,m], '-', linewidth=3, label= r'$\rho$='+str(rho))
plt.xlabel(r'$f_t$', fontsize=20)
plt.ylabel('PDF', fontsize=20)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.xscale('log')
# plt.yscale('log')
plt.savefig('ft-interevent-PDF.png', dpi=600, bbox_inches='tight')
plt.show()
