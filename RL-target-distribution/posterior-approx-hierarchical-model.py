import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import bernoulli
import itertools
from scipy.spatial import distance
import copy
import pickle
import matplotlib.pyplot as plt
import time
from scipy.stats import pareto
from scipy.stats import expon
from scipy.interpolate import NearestNDInterpolator
import random
from scipy.stats import uniform
from itertools import product
import networkx as nx
from random import sample
from scipy.stats import gamma



def sampling_targets():

    sampled_patches_x=norm.rvs(loc=5000, scale=2000, size=10 )
    sampled_patches_y=norm.rvs(loc=5000, scale=2000, size=10 )

    targets_x=[]
    targets_y=[]

    for t in range(10):

        add_targets_x=list(norm.rvs(loc=sampled_patches_x[t], scale=200, size=50 ))
        add_targets_y=list(norm.rvs(loc=sampled_patches_y[t], scale=200, size=50 ))
        targets_x=targets_x+add_targets_x
        targets_y=targets_y+add_targets_y


    #### for the example of visited states:

    sampled_patches_x=norm.rvs(loc=5000, scale=2000, size=10 )
    sampled_patches_y=norm.rvs(loc=5000, scale=2000, size=10 )

    targets_x_s=[]
    targets_y_s=[]


    for t in range(10):


        add_targets_x=list(norm.rvs(loc=sampled_patches_x[t], scale=200, size=50 ))
        add_targets_y=list(norm.rvs(loc=sampled_patches_y[t], scale=200, size=50 ))
        targets_x_s=targets_x_s+add_targets_x
        targets_y_s=targets_y_s+add_targets_y


    return (targets_x, targets_y)


targets_x, targets_y=sampling_targets()
targets_x_s, targets_y_s=sampling_targets()


plt.plot(targets_x_s,targets_y_s, 'o', color='blue')
plt.plot(targets_x,targets_y, 'o', color='red')
plt.plot
plt.show()


np.savetxt('targets_x.out', targets_x, delimiter=',')
np.savetxt('targets_y.out', targets_y, delimiter=',')


targets_x=np.loadtxt('targets_x.out')
targets_y=np.loadtxt('targets_y.out')


def target_detected_basic(x,y,ind):

    radius_to=distance.euclidean((x,y),(targets_x[ind], targets_y[ind]))

    if radius_to<=25:

        return 1

    else:

        return 0


#start with one set of data

zeros_x=[]
zeros_y=[]
ones_x=[[] for i in range(10)]
ones_y=[[] for i in range(10)]

label_list=500*[False]

for i in range(500):

    if target_detected_basic(targets_x_s[i], targets_y_s[i],i)==0:

        zeros_x.append(targets_x_s[i])
        zeros_y.append(targets_y_s[i])

    else:

        ind_p=int(np.floor(i/50))

        ones_x[ind_p].append(targets_x_s[i])
        ones_y[ind_p].append(targets_y_s[i])
        label_list[i] =True



    # creating subplot and figure


    # plt.plot(targets_x,targets_y, 'o')\
 #    plt.clf()
 #    plt.scatter(targets_x1, targets_y1,  color='blue')
 #    plt.scatter(targets_x, targets_y, color='red')
 #    plt.scatter(new_vals_x, new_vals_y, color='green')
 #    plt.pause(0.2)
 #
 #
 #    targets_x1=copy.deepcopy(new_vals_x)
 #    targets_y1 = copy.deepcopy(new_vals_y)
 #
 #
 # plt.show()




def test():

    S = 100
    tau_0_x=5000
    tau_0_y=5000
    sigma_y=200
    mu_0=5000
    k_0=1
    v_0=1

    for h in range(10):

        post_target_x = np.zeros((S, 500))
        post_target_y = np.zeros((S, 500))
        post_patch_x = np.zeros((S, 10))
        post_patch_y = np.zeros((S, 10))
        post_varp_x= np.zeros((S, 10))
        post_varp_y = np.zeros((S, 10))

        clase=1

        for s in range(S):

            print("iteration : " + str(s))

            for t in range(10):

                sub_targets_x_s = copy.deepcopy(targets_x_s[t * 50: t * 50 + 50])
                sub_targets_y_s = copy.deepcopy(targets_y_s[t * 50: t * 50 + 50])

                sub_targets_x_star = [norm.rvs(loc=i, scale=400 * clase + (1 - clase) * 50) for i in sub_targets_x_s]
                sub_targets_y_star = [norm.rvs(loc=i, scale=400 * clase + (1 - clase) * 50) for i in sub_targets_y_s]

                #sampling from conditional distribution the location of the patch
                # for x
                v_n = v_0 + 50
                k_n=k_0+50

                s_x=np.std(sub_targets_x_star)

                sigma_n2_x=(v_n**(-1))*(v_0*tau_0_x**2 + 49*s_x**2 + (k_n**(-1))*k_0*50*(np.mean(sub_targets_x_star)-mu_0)**2)
                beta=sigma_n2_x*0.5*v_n
                alpha=v_n*0.5
                inv_tau_02_x=gamma.rvs(a=alpha, scale=1/beta)
                tau_0_x=1/np.sqrt(inv_tau_02_x)
                b = mu_0 / (tau_0_x ** 2) + 50 * np.mean(sub_targets_x_star) / sigma_y ** 2
                a = tau_0_x ** (-2) + 50 * sigma_y ** (-2)
                mu_n = b / a
                tau_n_x = np.sqrt(1 / a)

                center_patch_x = norm.rvs(loc=mu_n, scale=tau_n_x)

                ## for y


                s_y = np.std(sub_targets_y_star)


                sigma_n2_y = (v_n ** (-1)) * (v_0 * tau_0_y ** 2 + 49 * s_y ** 2 + (k_n ** (-1)) * k_0 * 50 * (
                            np.mean(sub_targets_y_star) - mu_0) ** 2)

                beta = sigma_n2_y * 0.5 * v_n
                alpha = v_n * 0.5
                inv_tau_02_y = gamma.rvs(a=alpha, scale=1 / beta)
                tau_0_y = 1 / np.sqrt(inv_tau_02_y)

                b = mu_0 / (tau_0_y ** 2) + 50 * np.mean(sub_targets_y_star) / sigma_y ** 2
                a = tau_0_y ** (-2) + 50 * sigma_y ** (-2)
                mu_n = b / a
                tau_n_y = np.sqrt(1 / a)

                center_patch_y = norm.rvs(loc=mu_n, scale=tau_n_y)

                post_patch_x[s,t]=center_patch_x
                post_patch_y[s, t] = center_patch_y
                post_varp_x[s,t]=tau_n_y
                post_varp_y[s,t]=tau_0_y

                for j in range(50):

                    if label_list[t*50+j]:

                        continue



                    gap_x=np.array([max(k-sub_targets_x_star[j],0) for k in zeros_x])

                    gap_x_pos=gap_x[gap_x>0]
                    gap_x_up=gap_x_pos[gap_x_pos<=17.5]


                    if (len(gap_x_up)!=0):

                        continue

                    else:

                        sub_targets_x_s[j] = sub_targets_x_star[j]
                        sub_targets_y_s[j] = sub_targets_y_star[j]


                post_target_x[s, 50 * t:50 * t + 50] = sub_targets_x_s
                post_target_y[s, 50 * t:50 * t + 50] = sub_targets_y_s


        targets_x_s = np.mean(post_target_x, axis=0)
        targets_y_s = np.mean(post_target_y, axis=0)

        plt.clf()
        plt.scatter(targets_x_s, targets_y_s, color='blue')
        plt.scatter(targets_x, targets_y, color='red')
        plt.pause(0.2)

        for i in range(500):

            if target_detected_basic(targets_x_s[i], targets_y_s[i],i) == 0:

                zeros_x.append(targets_x_s[i])
                zeros_y.append(targets_y_s[i])

            else:

                ind_p = int(np.floor(i / 50))
                ones_x[ind_p].append(targets_x_s[i])
                ones_y[ind_p].append(targets_y_s[i])
                label_list[i] = True

    #
    plt.show()




