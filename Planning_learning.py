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

    def __init__(self,L,T,radius_detection):

        self.L = L
        self.T = T
        self.pos_x = L / 2
        self.pos_y = L / 2
        self.radius_detection = radius_detection
        self.a = np.sqrt(2)*self.radius_detection
        self.dim_table = int(np.ceil(self.L/self.a))
        self.alpha=0.01
        self.epsilon=0.1
        self.t=0
        self.n_planning=10

        # be careful with this parameter
        amplitude = np.arange(1,10)
        movement_index = []

        for A in amplitude:

            plus_i = A * np.array([-1, 0, 1])
            plus_j = A * np.array([-1, 0, 1])

            next_cell = list(product(plus_i, plus_j))
            next_cell.remove((0, 0))

            movement_index.append(next_cell)
            self.flat_list = [item for sublist in movement_index for item in sublist]

        dim_z = len(self.flat_list)
        self.table = np.random.rand(self.dim_table, self.dim_table, dim_z)

        keys = list(product(np.arange(self.dim_table),
                    np.arange(self.dim_table),
                            np.arange(dim_z)))

        values = [(0, 0)] * len(keys)
        self.model= dict(zip(keys, values))

        self.keys_sample = list(product(np.arange(self.dim_table),
                            np.arange(self.dim_table)))

        values_sample = [[]] * len(self.keys_sample)
        self.sample_dict=dict(zip(self.keys_sample, values_sample))
        self.visite_states=[]

        # my_dicty[(1, 1, 1)] = (1, 2)

    def epsilon_greedy_action(self,pos_x, pos_y):

        index_x = int(np.floor(pos_x / self.a))
        index_y = int(np.floor(pos_y / self.a))

        treshold=uniform.rvs(0,1)

        if treshold>=self.epsilon:

            index_z=np.argmax(self.table[index_x, index_y,:])

        else:

            index_z=np.random.choice(len(self.flat_list), 1)[0]

        return (index_x,index_y, index_z)

    def take_action(self,  triplet_index):

        self.sample_dict[(triplet_index[0],triplet_index[1])].append(triplet_index[2])
        self.visite_states.append((triplet_index[0],triplet_index[1]))
        delta_x=self.flat_list[triplet_index[2]][0]
        delta_y=self.flat_list[triplet_index[2]][1]

        new_index_x= triplet_index[0] + delta_x
        new_index_y= triplet_index[1] + delta_y

        if new_index_x<0:

            new_index_x=-1*new_index_x

        if new_index_x>self.dim_table:

            new_index_x=2*self.dim_table-new_index_x

        if new_index_y<0:

            new_index_y = -1 * new_index_y

        if new_index_y > self.dim_table:

            new_index_y = 2*self.dim_table - new_index_y


        new_pos_x = self.a * new_index_x
        new_pos_y=  self.a * new_index_y

        return (new_pos_x, new_pos_y)

    def update_state_agent(self, new_pos_x, new_pos_y):

        self.pos_x=new_pos_x
        self.pos_y=new_pos_y
        self.t+=1

    def update_q_values(self,triple_index, new_pos_x, new_pos_y, reward):

        index_x_prime = int(np.floor(new_pos_x / self.a))
        index_y_prime = int(np.floor(new_pos_y / self.a))

        self.table[triple_index[0],triple_index[1], triple_index[2]]=\
            self.table[triple_index[0],triple_index[1], triple_index[2]]+\
            self.alpha*(reward+max(self.table[index_x_prime, index_y_prime,:])-
                        self.table[triple_index[0], triple_index[1], triple_index[2]] )

    def update_model(self, triple_index, new_pos_x, new_pos_y, reward):

        index_x_prime = int(np.floor(new_pos_x / self.a))
        index_y_prime = int(np.floor(new_pos_y / self.a))

        self.model[triple_index]=(reward, index_x_prime,index_y_prime )


    def simulate_n_steps(self):

        for _ in range(self.n_planning):

            state= np.random.choice(self.visite_states, 1)[0]
            action= np.random.choice(self.sample_dict[state], 1)[0]

            reward,next_state=self.model[(state[0], state[1], action)]

    def reset_agent(self):

        self.pos_x = L / 2
        self.pos_y = L / 2
        self.t = 0

        self.t = 0

        # be careful with this parameter
        amplitude = np.arange(1, 10)
        movement_index = []

        for A in amplitude:
            plus_i = A * np.array([-1, 0, 1])
            plus_j = A * np.array([-1, 0, 1])

            next_cell = list(product(plus_i, plus_j))
            next_cell.remove((0, 0))

            movement_index.append(next_cell)
            self.flat_list = [item for sublist in movement_index for item in sublist]

        dim_z = len(self.flat_list)
        self.table = np.random.rand(self.dim_table, self.dim_table, dim_z)

        keys = list(product(np.arange(self.dim_table),
                            np.arange(self.dim_table),
                            np.arange(dim_z)))

        values = [(0, 0)] * len(keys)
        self.model = dict(zip(keys, values))

        self.keys_sample = list(product(np.arange(self.dim_table),
                                        np.arange(self.dim_table)))

        values_sample = [[]] * len(self.keys_sample)
        self.sample_dict = dict(zip(self.keys_sample, values_sample))
        self.visite_states = []


if __name__ == "__main__":

    L=10000
    T=20
    radius_detection=25
    a1=Agent(L,T, radius_detection)



a=[1,2,3,4,5]
b=[1,2,3,4,5]
c=[1,2,3]


keys=list(product(a,b,c))

values=[(0,0)]*len(keys)

my_dicty=dict(zip(keys, values))
my_dicty[(1,1,1)]=(1,2)




np.random.choice(a, 2)