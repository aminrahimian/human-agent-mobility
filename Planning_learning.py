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


# generate targets.

class Enviroment:
    def __init__(self, L, target_location):

        self.L = L
        self.target_location=target_location
        self.cont_prev=0
        self.cont_next=0

    def update_target_status(self, key_dict):

        new_tuple = (self.targets[key_dict][0],self.targets[key_dict][1],1)
        self.targets[key_dict] = new_tuple

    def update_threat_status(self, key_dict):

        new_tuple = (self.threats[key_dict][0], self.threats[key_dict][1], 1)
        self.threats[key_dict] = new_tuple

    def reset_target_status(self):

        for t in range(self.n_targets):
            self.targets[t]=(self.targets[t][0],self.targets[t][1],0)

        for t in range(self.n_threats):
            self.threats[t] = (self.threats[t][0], self.threats[t][1], 0)

    def collected_targets(self, pos_x, pos_y, radius_detection):

        """ Return the number of targets given the historical
        position of the agent
        """
        initial_range = len(self.target_location)
        agent_position=(pos_x,pos_y)
        agent_position_list=[(agent_position)]
        list_target_location = list(self.target_location)
        itertools.product(agent_position_list, list_target_location)
        list_distances=np.array([distance.euclidean(agent_position, list_target_location[i]) for i in range(initial_range)])
        cont= np.sum(list_distances<=radius_detection)
        # print(" cont " + str(cont))
        self.cont_prev+=cont

        if cont>0:

            R=10*cont

        else:

            R=-1

        return (R, cont)

    def reset_enviroment(self,target_location):

        self.target_location = target_location
        self.cont_prev = 0



class Agent:

    def __int__(self,L,T,radius_detection):

        self.L = L
        self.T = T
        self.pos_x = L / 2
        self.pos_y = L / 2
        self.radius_detection = radius_detection
        self.a = np.sqrt(2)*self.radius_detection
        self.dim_table = int(np.ceil(self.L/self.a))
        self.table = np.ones((self.dim_table,self.dim_table,2))
        self.table[:, :, 0] = self.table[:, :, 0] * 0.11
        self.table[:, :, 1] = self.table[:, :, 1] * 0.10
        self.alpha=0.01
        self.triplet_obs_states=[]


    def mu_epsilon_greedy(self,pos_x, pos_y):

        index_x=int(np.floor(pos_x/self.a))
        index_y=int(np.floor(pos_y/self.a))

        self.table[index_x,index_y]

        if uniform.rvs()>self.epsilon:

            mu=np.argmax(self.table[index_x,index_y,:]) + 2

        else:

            mu=np.random.choice(2,1)[0] +2

        return  mu


    def sample_step_length(self, mu):

        alpha = mu - 1
        V = np.random.uniform(-np.pi * 0.5, 0.5 * np.pi, 1)[0]
        W = expon.rvs(size=1)[0]
        cop1 = (np.sin(alpha * V)) / ((np.cos(V)) ** (1 / alpha))
        cop2 = ((np.cos((1 - alpha) * V)) / W) ** (((1 - alpha) / alpha))
        elle = cop1 * cop2

        if elle <= 0:

            elle = 10 * min(500, -1 * elle)
            step_l = max(25, elle)

        else:

            elle = 10 * min(500, elle)
            step_l = max(25, elle)

        # print("Taking this value:  " + str(elle))

        # angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # print("angle vel :" + str(angle))
        # vel_x = 10*elle * np.cos(angle)
        # vel_y = 10*elle * np.sin(angle)

        return step_l







H=np.ones((4,4,2))

