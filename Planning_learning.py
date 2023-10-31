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

# generate targets.
class Enviroment:

    def __init__(self, L, target_location,radius_detection):

        self.L = L
        self.target_location=target_location
        self.cont_prev=0
        self.cont_next=0
        self.radius_detection = radius_detection


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

    def collected_targets(self, pos_x, pos_y):

        """ Return the number of targets given the historical
        position of the agent
        """

        list_target_location=copy.deepcopy(self.target_location)
        to_delete = []

        for element in list_target_location:

            segment_length = distance.euclidean((pos_x, pos_y), element)

            if segment_length <= self.radius_detection:

                to_delete.append(element)


        for i in to_delete:

            list_target_location.remove(i)

        self.target_location=list_target_location

        if len(to_delete)>0:

            print(" * * * *  * Find a target at Location " +str((pos_x, pos_y)))
            # time.sleep(5)
            R=10*len(to_delete)

        else:

            R=-1

        return (R, len(to_delete))


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
        self.alpha=5e-4
        self.epsilon=0.1
        self.t=0
        self.n_planning=100

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
        # self.table = np.random.rand(self.dim_table, self.dim_table, dim_z)
        self.table =np.zeros((self.dim_table, self.dim_table, dim_z))

        keys = list(product(np.arange(self.dim_table),
                    np.arange(self.dim_table),
                            np.arange(dim_z)))

        values = [(0, 0,0)] * len(keys)
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

        self.sample_dict[(triplet_index[0],triplet_index[1])]=self.sample_dict[(triplet_index[0],triplet_index[1])]+ [triplet_index[2]]

        if not (triplet_index[0],triplet_index[1]) in self.visite_states:

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

        # print("Moving to :"  +str((new_pos_x, new_pos_y)))

        self.pos_x=new_pos_x
        self.pos_y=new_pos_y
        self.t+=1

    def update_q_values(self,triplet_index, new_pos_x, new_pos_y, reward):

        index_x_prime = int(np.floor(new_pos_x / self.a))
        index_y_prime = int(np.floor(new_pos_y / self.a))

        # print("Previous value: " +  str(self.table[triplet_index[0],triplet_index[1], triplet_index[2]] ))
        self.table[triplet_index[0],triplet_index[1], triplet_index[2]]=\
            self.table[triplet_index[0],triplet_index[1], triplet_index[2]]+\
            self.alpha*(reward+max(self.table[index_x_prime, index_y_prime,:])-
                        self.table[triplet_index[0], triplet_index[1], triplet_index[2]] )

        # print("Updated value :" + str(self.table[triplet_index[0],triplet_index[1], triplet_index[2]] ))

    def update_model(self, triple_index, new_pos_x, new_pos_y, reward):

        index_x_prime = int(np.floor(new_pos_x / self.a))
        index_y_prime = int(np.floor(new_pos_y / self.a))

        self.model[triple_index]=(reward, index_x_prime,index_y_prime )

    def simulate_n_steps(self):

        for k in range(self.n_planning):

            # print(str(k))
            # print("Visited states: "  +str(self.visite_states))
            state= sample(self.visite_states, 1)[0]
            # print("Sample state: " + str(state))
            action= sample(self.sample_dict[state], 1)[0]
            # print("Chosen actions " + str(self.sample_dict[state]))
            # print("sample action" +str(action))
            reward,next_state_x, next_state_y=self.model[(state[0], state[1], action)]

            # print("Previous q value : " +str(self.table[state[0], state[1], action]))

            self.table[state[0], state[1], action] = \
                self.table[state[0], state[1], action] + \
                self.alpha * (reward + max(self.table[next_state_x, next_state_y, :]) -
                              self.table[state[0], state[1], action])

            # print("Updated q value : " + str(self.table[state[0], state[1], action]))

    def reset_agent(self):

        self.pos_x = L / 2
        self.pos_y = L / 2
        self.t = 0

    def save_data(self):

        np.savetxt('./planning_learning_data/visited_states.csv', self.visite_states, delimiter=',')

        with open('./planning_learning_data/tabular.pkl', 'wb') as file:
            pickle.dump(self.model, file)


        with open('./planning_learning_data/model.pkl', 'wb') as file:
            pickle.dump(self.model, file)

        with open('./planning_learning_data/sample_dict.pkl', 'wb') as file:
            pickle.dump(self.sample_dict, file)

    def load_data(self):

        self.visite_states=list(np.loadtxt('./planning_learning_data/visited_states.csv', delimiter=','))

        with open('./planning_learning_data/tabular.pkl', 'rb') as file:
            # Call load method to deserialze
            self.model = pickle.load(file)

        with open('./planning_learning_data/model.pkl', 'rb') as file:
            # Call load method to deserialze
            self.model = pickle.load(file)

        with open('./planning_learning_data/sample_dict.pkl', 'rb') as file:
            # Call load method to deserialze
            self.sample_dict = pickle.load(file)


if __name__ == "__main__":

    L=10000
    T=20
    radius_detection=25
    a1=Agent(L,T, radius_detection)

    learning=[]

    targets = np.loadtxt('target_large.csv', delimiter=',')
    target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]
    enviroment = Enviroment(L, target_location, radius_detection)


    for nn in range(1000):

        targets = np.loadtxt('target_large.csv', delimiter=',')
        target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]
        enviroment = Enviroment(L, target_location, radius_detection)

        print( "---------episode ---" + str(nn))


        for t in range(5000):

            pos_x=a1.pos_x
            pos_y=a1.pos_y
            triplet_index=a1.epsilon_greedy_action(pos_x, pos_y)
            new_pos=a1.take_action(triplet_index)
            a1.update_state_agent(new_pos[0], new_pos[1])
            reward, cont = enviroment.collected_targets(new_pos[0], new_pos[1])
            a1.update_q_values(triplet_index, new_pos[0], new_pos[1], reward)
            a1.update_model(triplet_index, new_pos[0], new_pos[1], reward)
            a1.simulate_n_steps()
            # time.sleep(1)

        print("targets leftovers " + str(len(enviroment.target_location)))

        learning.append(500- len(enviroment.target_location))
        a1.reset_agent()


    a1.save_data()


name_file='./planning_learning_data/learning_' + str(a1.n_planning)+'.csv'
np.savetxt(name_file, learning, delimiter=',')

