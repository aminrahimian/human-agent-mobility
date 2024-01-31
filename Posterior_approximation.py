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



sampled_patches_x=norm.rvs(loc=5000, scale=2200, size=10 )
sampled_patches_y=norm.rvs(loc=5000, scale=2200, size=10 )


targets_x=[]
targets_y=[]


for t in range(10):


    add_targets_x=list(norm.rvs(loc=sampled_patches_x[t], scale=200, size=50 ))
    add_targets_y=list(norm.rvs(loc=sampled_patches_y[t], scale=200, size=50 ))
    targets_x=targets_x+add_targets_x
    targets_y=targets_y+add_targets_y


#### for the example of visited states:

sampled_patches_x=norm.rvs(loc=5000, scale=2200, size=10 )
sampled_patches_y=norm.rvs(loc=5000, scale=2200, size=10 )

targets_x_s=[]
targets_y_s=[]


for t in range(10):


    add_targets_x=list(norm.rvs(loc=sampled_patches_x[t], scale=300, size=50 ))
    add_targets_y=list(norm.rvs(loc=sampled_patches_y[t], scale=300, size=50 ))
    targets_x_s=targets_x_s+add_targets_x
    targets_y_s=targets_y_s+add_targets_y



# plt.plot(targets_x,targets_y, 'o')
plt.plot(targets_x_s,targets_y_s, 'o', color='blue')
plt.plot(targets_x,targets_y, 'o', color='red')
plt.plot
plt.show()


def target_detected_basic(x,y):


    ref_point= 500*[(x,y)]
    targets=[(targets_x[i], targets_y[i])  for i in range(500)]
    radius_to=np.array([distance.euclidean(ref_point[t], targets[t]) for t in range(500)])

    if np.sum(radius_to<=25)==0:

        return 0

    else:

        return 1



#start with one set of data

zeros_x=[]
zeros_y=[]
ones_x=[]
ones_y=[]


for i in range(500):

    if target_detected_basic(targets_x_s[i], targets_y_s[i])==0:

        zeros_x.append(targets_x_s[i])
        zeros_y.append(targets_y_s[i])

    else:

        ones_x.append(targets_x_s[i])
        ones_y.append(targets_y_s[i])





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


for h in range(1):

    post_target_x=np.zeros((100,500))
    post_target_y=np.zeros((100,500))
    post_patch_x=np.zeros((100,10))
    post_patch_y= np.zeros((100, 10))

    for s in range(100):


        for t in range(10):

            sub_targets_x_s = copy.deepcopy(targets_x_s[t * 50: t * 50 + 50])
            sub_targets_y_s = copy.deepcopy(targets_y_s[t * 50: t * 50 + 50])

            sub_targets_x_star=[norm.rvs(loc=i, scale=400) for i in sub_targets_x_s]
            sub_targets_y_star=[norm.rvs(loc=i, scale=400) for i in sub_targets_y_s]

            likelihood=0

            for j in range(50):

                distances=  np.array([distance.euclidean((zeros_x[i], zeros_y[i]),(sub_targets_x_star[j], sub_targets_y_star[j])) for i in range(len(zeros_x))])
                bool_distances=distances>=25
                bin_distances=bool_distances.astype(int).reshape((1,len(zeros_x)))
                displacement=0.01*np.ones((1,len(zeros_x)))
                likelihood +=np.sum(np.log(np.add(bin_distances, displacement)))

                distances = np.array([distance.euclidean((zeros_x[i], zeros_y[i]),
                                                         (sub_targets_x_s[j], sub_targets_y_s[j])) for i in
                                      range(len(zeros_x))])
                bool_distances = distances >= 25
                bin_distances = bool_distances.astype(int).reshape((1,len(zeros_x)))
                displacement = 0.01 * np.ones((1, len(zeros_x)))

                likelihood -= np.sum(np.log(np.add(bin_distances, displacement)))

                likelihood +=np.log(norm.pdf(sub_targets_x_star[j],loc=sampled_patches_x[t], scale=200))- \
                              np.log(norm.pdf(sub_targets_x_s[j], loc=sampled_patches_x[t], scale=200))+ \
                              np.log(norm.pdf(sub_targets_y_star[j], loc=sampled_patches_y[t], scale=200)) - \
                              np.log(norm.pdf(sub_targets_y_s[j], loc=sampled_patches_y[t], scale=200))



                u=np.log(uniform.rvs())


                if u<likelihood:

                   sub_targets_x_s[j]= sub_targets_x_star[j]
                   sub_targets_y_s[j] = sub_targets_y_star[j]

                else:

                    pass


                post_target_x[s,50*t:50*t+50]=sub_targets_x_s
                post_target_y[s,50*t:50*t+50]=sub_targets_y_s


            sampled_patches_x_star=[norm.rvs(loc=i, scale=500) for i in sampled_patches_x]
            sampled_patches_y_star = [norm.rvs(loc=i, scale=500) for i in sampled_patches_y]

            likelihood = 0

            for j in range(50):


                likelihood += np.log(norm.pdf(post_target_x[s,50*t:50*t+50][j], loc=sampled_patches_x_star[t], scale=200)) + \
                              np.log(norm.pdf(post_target_y[s, 50 * t:50 * t + 50][j], loc=sampled_patches_y_star[t],
                                              scale=200))

                likelihood -= np.log(
                    norm.pdf(post_target_x[s, 50 * t:50 * t + 50][j], loc=sampled_patches_x[t], scale=200)) + \
                              np.log(norm.pdf(post_target_y[s, 50 * t:50 * t + 50][j], loc=sampled_patches_y[t],
                                              scale=200))


            likelihood+= np.log(norm.pdf(sampled_patches_x_star[t], loc=5000, scale=3000))+ \
                         np.log(norm.pdf(sampled_patches_y_star[t], loc=5000, scale=3000)) -\
            np.log(norm.pdf(sampled_patches_x[t], loc=5000, scale=3000))- \
            np.log(norm.pdf(sampled_patches_y[t], loc=5000, scale=3000))

            u = np.log(uniform.rvs())

            if u < likelihood:

                sampled_patches_x[t] = sampled_patches_x_star[t]
                sampled_patches_y[t] = sampled_patches_x_star[t]

            else:

                pass

            post_patch_x[s, t] = sampled_patches_x[t]
            post_patch_y[s, t] = sampled_patches_y[t]






plt.plot(post_target_x[9,:],post_target_y[9,:], 'o')
plt.plot(targets_x_s,targets_y_s, 'o', color='blue')
plt.plot(targets_x,targets_y, 'o', color='red')
plt.plot
plt.show()



