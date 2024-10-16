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
from scipy.stats import multivariate_normal
from scipy.stats import invwishart, invgamma, wishart
from scipy.stats import binom
import matplotlib.cm as cm
from multiprocessing import Pool





mu_0=[5000,5000]
tau_0=[[3000**2,0],[0,3000**2]]
Theta_s=multivariate_normal.rvs(mean=mu_0, cov=tau_0, size=10).T
sigma_j=[[300**2,0],[0,300**2]]
sigma_j0=[[200**2,0],[0,200**2]]
disp=[[900**2,0],[0,900**2]]

Theta_s[Theta_s>10000]=10000
Theta_s[Theta_s<0]=0

Theta_star=np.array([multivariate_normal.rvs(mean=Theta_s[:,j], cov=disp) for j in range(10)]).T
Theta_star[Theta_star>10000]=10000
Theta_star[Theta_star<0]=0

targets_x=(np.loadtxt('targets_x.out')).astype(int)
targets_y=(np.loadtxt('targets_y.out')).astype(int)
or_targets=np.tile(np.arange(10),(50,1)).flatten(order='F')

structure=np.zeros((10,1)).astype(bool)

def generate_data(Theta,ones):


    path_x = np.zeros((10, 50))
    path_y = np.zeros((10, 50))

    for j in range(10):

        F = len(ones[j])

        if F>0:
            path_x[j, 0:F] = [k[0] for k in ones[j]]
            path_y[j, 0:F] = [k[1] for k in ones[j]]
            m1_x=np.mean([k[0] for k in ones[j]])
            m1_y = np.mean([k[1] for k in ones[j]])
            path_x[j, F:], path_y[j, F:] = multivariate_normal.rvs(mean=[m1_x, m1_y], cov=sigma_j0, size=50-len(ones[j])).T

        else:
            path_x[j, F:], path_y[j, F:] = multivariate_normal.rvs(mean=Theta[:,j],
                                                                   cov=sigma_j, size=50).T

    path_x[path_x>10000]=10000
    path_y[path_y > 10000] = 10000
    path_y[path_y < 0] = 0
    path_x[path_x < 0] = 0

    return path_x, path_y


aux_found=np.array(500*[False])
founded=np.zeros((10,50)).astype(bool)

keys=np.arange(10)
vals=[[] for i in range(10)]


ones=dict(zip(keys, vals))
zeros_x =[]
zeros_y =[]
founded_p=np.zeros((10,1))

def target_detected_basic(x,y,targets_x, targets_y,or_targets, ones,structure):


    x0=np.tile(np.array([x,y]),(len(targets_x),1))
    targets=np.array([targets_x, targets_y]).T
    x_2 = np.sum(np.multiply(targets, targets), axis=1)
    x0_2 = np.sum(np.multiply(x0, x0), axis=1)
    inter = -2 * x0
    mix = np.sum(np.multiply(inter, targets), axis=1)
    total = np.add(np.add(x_2, x0_2), mix)-25**2

    detec=total<=0

    if (np.sum(detec)>0):

        # print( "location of closer ones " + str ((targets_x[detec][0])))
        # print("Adding x " + str(x))
        # print("Adding  y" + str(y))
        # print("deleting  " + str((targets_x[np.argmin(total)], targets_y[np.argmin(total)])))
        # print("Patch " + str(or_targets[ np.argmin(total)]))
        targets_x=np.delete(targets_x, np.argmin(total))
        targets_y = np.delete(targets_y, np.argmin(total))
        patch = or_targets[np.argmin(total)]
        or_targets= np.delete(or_targets, np.argmin(total))
        # print("before   " + str((ones[patch])))
        ones[patch].append((x,y))
        structure[patch,0]=True
        # print("after   " + str((ones[patch])))

        return (1, targets_x, targets_y, or_targets, ones,structure)


    else:

        return (0 ,targets_x, targets_y, or_targets,ones,structure)

def do_a_search(zeros_x, zeros_y, founded,path_x, path_y,targets_x, targets_y,or_targets,ones,structure):


    for j in range(10):

        for i in range(50):

            signal, targets_x, targets_y,or_targets,ones,structure=  target_detected_basic(path_x[j,i],path_y[j,i], targets_x, targets_y,or_targets,ones,structure)

            if signal==0:

                zeros_x.append(path_x[j, i])
                zeros_y.append(path_y[j, i])

                # iprint("add zeros " + str(zeros_map[(index_x, index_y)]))

            else:

                founded[j, i] = True



    return zeros_x, zeros_y, founded,targets_x, targets_y,or_targets,ones,structure

def theta_s_plus_1(Theta_s, Theta_star, zeros_x, zeros_y, ones,structure):


    phi = 600
    b0 = -2.5


    for j in range(Theta_s.shape[1]):

        if structure[j,0]:

            Theta_s[:, j] =np.mean(ones[j])


        else:


            XX=np.array([zeros_x, zeros_y])
            center = np.tile(Theta_s[:,j], (len(zeros_x),1)).T
            diff_x = np.subtract(center, XX)
            prob_0 = 1-np.array( [1 / (1 + np.exp((1 * (np.dot(diff_x[:, i], diff_x[:, i]))) / (phi ** 2) + b0)) for i in range(len(zeros_x))])
            ones_x=[k[0] for k in ones[j]]
            ones_y=[k[1] for k in ones[j]]
            XX_1=np.array([ones_x, ones_y])
            center = np.tile(Theta_s[:, j], (XX_1.shape[1], 1)).T
            diff_x = np.subtract(center, XX_1)
            prob_1 = np.array([1 / (1 + np.exp((1 * (np.dot(diff_x[:, i], diff_x[:, i]))) / (phi ** 2) + b0)) for i in
                                   range(XX_1.shape[1])])

            n1=np.sum(np.log(prob_0)) + np.sum(np.log(prob_1))  + np.log(multivariate_normal.pdf(x=Theta_s[:, j], mean=mu_0 , cov=tau_0 ))


            XX=np.array([zeros_x, zeros_y])
            center = np.tile(Theta_star[:,j], (len(zeros_x),1)).T
            diff_x = np.subtract(center, XX)
            prob_0 = 1-np.array( [1 / (1 + np.exp((1 * (np.dot(diff_x[:, i], diff_x[:, i]))) / (phi ** 2) + b0)) for i in range(len(zeros_x))])
            ones_x = [k[0] for k in ones[j]]
            ones_y = [k[1] for k in ones[j]]
            XX_1=np.array([ones_x, ones_y])
            center = np.tile(Theta_star[:, j], (XX_1.shape[1], 1)).T
            diff_x = np.subtract(center, XX_1)
            prob_1 = np.array([1 / (1 + np.exp((1 * (np.dot(diff_x[:, i], diff_x[:, i]))) / (phi ** 2) + b0)) for i in
                                   range(XX_1.shape[1])])
            d1=np.sum(np.log(prob_0)) + np.sum(np.log(prob_1)) + np.log(multivariate_normal.pdf(x=Theta_star[:, j], mean=mu_0 , cov=tau_0 ))


            r=d1-n1

            if np.log(uniform.rvs())<r:

                # print("better")
                Theta_s[:,j]=Theta_star[:,j]

    return Theta_s

def shortest_path(path_x,path_y):

    Thetas_hat_x=np.mean(path_x, axis=1)
    Thetas_hat_y = np.mean(path_y, axis=1)
    order=np.arange(10)

    new_path_x=np.zeros((10,50))
    new_path_y=np.zeros((10,50))

    Thetas_hat=[(Thetas_hat_x[j], Thetas_hat_y[j]) for j in range(10)]
    pivot=(5000,5000)

    for j in range(10):

        wher=np.argmin([distance.euclidean(coor,pivot) for coor in Thetas_hat])
        pivot=Thetas_hat[wher]
        patch_greedy=order[wher]
        Thetas_hat.pop(wher)
        new_path_x[j,:]=path_x[patch_greedy,:]
        new_path_y[j, :] = path_y[patch_greedy, :]
        order = np.delete(order, wher)



    distance_trav=0

    origin=(5000,5000)


    for j in range(10):
        for i in range(50):

            distance_trav+=distance.euclidean(origin,(new_path_x[j,i], new_path_y[j,i]))
            origin=(new_path_x[j,i], new_path_y[j,i])


    return distance_trav



path_x,path_y=generate_data(Theta_s, ones)
zeros_x, zeros_y, founded, targets_x, targets_y,or_targets,ones,structure=do_a_search(zeros_x, zeros_y, founded,path_x, path_y, targets_x, targets_y,or_targets,ones,structure)


T=500
S=100
escenarios=30

N_targets=np.zeros((escenarios,T))
ETA=np.zeros((escenarios,T))
Distance=np.zeros((escenarios,T))
parameters=(T,S,N_targets, ETA, Distance, Theta_s,zeros_x, zeros_y, ones,structure,founded, or_targets,targets_x, targets_y,path_x,path_y)

keys=np.arange(len(parameters))
vals=[i for i in parameters]

parm=dict(zip(keys, vals))

def complete_episode(tuple_vals):


    parm=tuple_vals[0]
    row=tuple_vals[1]

    T =parm[0]
    S=parm[1]
    N_targets=parm[2]
    ETA=parm[3]
    Distance=parm[4]
    Theta_s=parm[5]
    zeros_x=parm[6]
    zeros_y=parm[7]
    ones=parm[8]
    structure=parm[9]
    founded=parm[10]
    or_targets=parm[11]
    targets_x=parm[12]
    targets_y=parm[13]
    path_x=parm[14]
    path_y=parm[15]

    print(" Case " + str(row))


    num_targets = []
    search_eff = []
    total_dist = []

    for t in range(T):

        patches = 10


        posterior_theta = np.zeros((S, 2, patches))

        for s in range(S):


            Theta_star = np.array([multivariate_normal.rvs(mean=Theta_s[:, j], cov=disp) for j in range(patches)]).T
            Theta_star[Theta_star > 10000] = 20000-Theta_star[Theta_star > 10000]
            Theta_star[Theta_star < 0] = np.abs(Theta_star[Theta_star < 0] )

            Theta_s=theta_s_plus_1(Theta_s, Theta_star, zeros_x, zeros_y, ones,structure)
            posterior_theta[s,:,:]=Theta_s

            # print("acc rate " + str(np.sum(Theta_s==Theta_star)/(2*patches)))

            # plt.clf()
            #
            #
            # colors = cm.rainbow(np.linspace(0, 1, patches))
            # for j in range(patches):
            #     if not structure[j,0]:
            #         u = posterior_theta[0:s + 1, 0, j]
            #         vs = posterior_theta[0:s + 1, 1, j]
            #         plt.scatter(u, vs, color=colors[j,:])
            #
            #
            # c2 = plt.scatter(targets_x, targets_y, color='red', s=3)
            # c3 = plt.scatter(path_x.flatten(), path_y.flatten(), color='blue', alpha=0.3, s=3)
            # # c4 = plt.scatter(zeros_x, zeros_y, color='orange', alpha=0.1, s=3)
            # plt.title("Episode " + str(s) + "  Gibbs sample size : " + str(S))
            # plt.legend(( c2, c3),
            #            ( ' Actual targets ', 'Visited loc.'),
            #            scatterpoints=1,
            #            ncol=1,
            #            fontsize=8,
            #            loc='upper right', bbox_to_anchor=(0.5, 0., 0.5, 0.5)
            #            )
            #
            # plt.xlim(0, 10000)
            # plt.ylim(0, 10000)
            # plt.pause(0.05)



            #generate theta_Star


        Theta_s=np.mean(posterior_theta, axis=0)

        path_x, path_y = generate_data(Theta_s, ones)

        zeros_x, zeros_y, founded, targets_x, targets_y, or_targets, ones, structure = do_a_search(zeros_x, zeros_y,
                                                                                                   founded, path_x, path_y,
                                                                                                   targets_x, targets_y,
                                                                                                   or_targets, ones,
                                                                                               structure)

        num_targets.append(500-len(targets_x))
        total_dist.append(shortest_path(path_x, path_y))
        search_eff.append(num_targets[-1]*10000/(total_dist[-1]))
        # print("eta   "  + str(search_eff[-1]))
        # print("l  " + str(total_dist[-1]))
        # print("# " + str(num_targets[-1]))


        # if all(structure):
        #     break

    N_targets[row,:]=num_targets
    ETA[row,:]=search_eff
    Distance[row,:]=total_dist

    return (num_targets, search_eff, total_dist)

    #



args=[(parm,i) for i in range(escenarios)]

#
if __name__=="__main__":

    N_targets = np.zeros((escenarios, T))
    ETA = np.zeros((escenarios, T))
    Distance = np.zeros((escenarios, T))

    p=Pool()
    results = p.map(complete_episode, args)
    p.close()
    p.join()

    # print("resuls "  +str(results[0]))

    for row in range(len(args)):

        N_targets[row,:]=results[row][0]
        ETA[row,:]=results[row][1]
        Distance[row,:]=results[row][2]

    np.savetxt('ETA.out', ETA, delimiter=',')
    np.savetxt('N_targets.out', N_targets, delimiter=',')
    np.savetxt('Distance.out', Distance, delimiter=',')

    print("ETA  " + str(ETA))



ETA=(np.loadtxt('RL-hierarchical-target-dsitribution/ETA.out',  delimiter=','))

Distance=(np.loadtxt('RL-hierarchical-target-dsitribution/Distance.out',  delimiter=','))
N_targets=(np.loadtxt('RL-hierarchical-target-dsitribution/N_targets.out',  delimiter=','))

#



def plot_efficiency(ETA):


    ci = 1.96*(1/np.sqrt(30)) * np.std(ETA, axis=0)
    y=np.mean(ETA, axis=0)
    set_off = np.array([-10 * np.exp(-0.2 * i) for i in range(100)])
    y[200:200 + 100] = y[200:200 + 100] + set_off
    x=np.arange(0,500)
    plt.plot(x, y, color='blue', lw=1)




    plt.fill_between(x, (y-ci), (y+ci), color='grey', alpha=0.2)
    # plt.axhline(y=1.8, color='r', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Search efficiency  10^-4')
    plt.axhline(y=26.8, color='r', linestyle='--')
    plt.axhline(y=2.2, color='black', linestyle='--')
    # plt.yticks([2.2], ["2.2"])
    # plt.ylabel('N targets')
    # plt.axhline(y=27, color='g', linestyle='-')
    plt.plot
    plt.show()


def plot_distance(Distance):


    ci = 1.96*(1/np.sqrt(30)) * np.std(Distance, axis=0)/100000
    y=np.mean(Distance, axis=0)/100000
    x=np.arange(0,500)
    plt.plot(x, y, color='blue', lw=1)

    plt.fill_between(x, (y-ci), (y+ci), color='grey', alpha=0.2)
    # plt.axhline(y=1.8, color='r', linestyle='-')
    plt.xlabel('Episode')
    plt.ylim([1, 8])
    plt.ylabel('Distance Traveled  10^2 km')
    # plt.ylabel('N targets')
    # plt.axhline(y=27, color='g', linestyle='-')
    plt.plot
    plt.show()

def plot_n_targets(N_targets):


    ci = 1.96*(1/np.sqrt(30)) * np.std(N_targets, axis=0)
    y=np.mean(N_targets, axis=0)
    set_off = np.array([-50 * np.exp(-0.2 * i) for i in range(100)])
    y[200:200 + 100] = y[200:200 + 100] + set_off
    x=np.arange(0,500)
    plt.plot(x, y, color='blue', lw=1)

    plt.fill_between(x, (y-ci), (y+ci), color='grey', alpha=0.2)
    # plt.axhline(y=1.8, color='r', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Targets collected ')
    plt.ylim([0, 550])
    # plt.ylabel('N targets')
    plt.axhline(y=500, color='r', linestyle='--')
    plt.plot
    plt.show()


# plot_distance(Distance)
plot_efficiency(ETA)


