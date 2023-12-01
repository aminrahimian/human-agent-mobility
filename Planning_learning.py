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
        target_close=np.array([distance.euclidean((pos_x, pos_y),i) for i in self.target_location])
        n_targets_detected=np.sum(target_close<= self.radius_detection)

        # list_target_location=copy.deepcopy(self.target_location)
        # to_delete = []
        #
        # for element in list_target_location:
        #
        #     segment_length = distance.euclidean((pos_x, pos_y), element)
        #
        #     if segment_length <= self.radius_detection:
        #
        #         to_delete.append(element)

        #
        # for i in to_delete:
        #
        #     list_target_location.remove(i)
        #
        # self.target_location=list_target_location

        if n_targets_detected>0:

            print(" * * * *  * Find a target at Location " +str((pos_x, pos_y)))
            # time.sleep(5)
            R=n_targets_detected

        else:

            R=0

        return (R,n_targets_detected)


    def reset_enviroment(self,target_location):

        self.target_location = target_location
        self.cont_prev = 0


class Agent:

    def __init__(self,L,T,radius_detection):

        self.L = L
        self.T = T
        self.pos_x = []
        self.pos_y = []
        self.radius_detection = radius_detection
        self.a = np.sqrt(2)*self.radius_detection
        self.dim_table = (int(np.ceil(self.L/self.a)))
        self.alpha=0.1
        self.epsilon=0.1
        self.t=0
        self.n_planning=10


        # be careful with this parameter

        # list of x and y

        self.position_nodes=[]

        for i in range(self.dim_table):
            for j in range(self.dim_table):

                x_i = self.a*i +self.a*0.5
                y_i = self.a * j + self.a * 0.5
                self.position_nodes.append((x_i, y_i))


        loc_center=[distance.euclidean((L/2,L/2),i) for i in self.position_nodes]

        self.current_node=np.argmin(loc_center)
        self.nodes=np.arange(len(self.position_nodes))

        # self.table = np.random.rand(self.dim_table, self.dim_table, dim_z)
        self.table =np.zeros((len(self.position_nodes),len(self.position_nodes)))


        self.model= []
        self.enviroment=[]

        self.keys_sample = list(product(np.arange(self.dim_table),
                            np.arange(self.dim_table)))

        values_sample = [[]] * len(self.keys_sample)
        self.sample_dict=dict(zip(self.keys_sample, values_sample))
        self.visite_states=[]

        # my_dicty[(1, 1, 1)] = (1, 2)

    def deterministic_model(self, target_location):

        new_target_location=[]

        for target in target_location:

            new_x=target[0]+norm.rvs(0,30,1)[0]
            new_y = target[1] + norm.rvs(0, 30, 1)[0]

            new_target_location.append((new_x,new_y))

        model=[]

        for i in self.nodes:

            distances=np.array([distance.euclidean((self.position_nodes[i][0],self.position_nodes[i][1]),(j[0],j[1])) for j in new_target_location])
            model.append(np.sum(distances<=self.radius_detection))
            print("node " +str(i)  + " with targets"+ str(np.sum(distances<=self.radius_detection)))


        self.model=model

    def actual_model(self, target_location):

        enviroment=[]

        for i in self.nodes:
            distances = np.array(
                [distance.euclidean((self.position_nodes[i][0], self.position_nodes[i][1]), (j[0], j[1])) for j in
                 target_location])
            enviroment.append(np.sum(distances <= self.radius_detection))

            print("node actual model " + str(i) + " with targets" + str(np.sum(distances <= self.radius_detection)))

        self.enviroment = enviroment

    def epsilon_greedy_action(self,node):

        treshold=uniform.rvs(0,1)
        self.table[node, node]=-1000

        if treshold>=self.epsilon:

            next_node=np.argmax(self.table[node, :])

        else:

            next_node=np.random.choice(self.nodes)

            if next_node==node:

                try:

                    next_node+=1

                except:

                    next_node-=1

        return next_node

    def update_state_agent(self, next_node):

        # print("Moving to :"  +str((new_pos_x, new_pos_y)))

        self.current_node=next_node
        self.pos_x.append(self.position_nodes[next_node][0])
        self.pos_y.append(self.position_nodes[next_node][1])
        self.t+=1

    def update_q_values(self,current_node,next_node, reward):


        # print("Previous value: " +  str(self.table[triplet_index[0],triplet_index[1], triplet_index[2]] ))
        self.table[current_node, next_node]=\
            self.table[current_node, next_node]+\
            self.alpha*(reward+max(self.table[next_node,:])-
                        self.table[current_node, next_node] )

        # print("Updated value :" + str(self.table[triplet_index[0],triplet_index[1], triplet_index[2]] ))

    def update_model(self, next_node, reward):

        self.model[next_node]=reward

    def simulate_n_steps(self):

        for k in range(self.n_planning):

            print("Simulating planning " + str(k))

            for t in range(100):

                current_node = self.current_node
                next_node = self.epsilon_greedy_action(current_node)
                # print("Going to " + str((self.position_nodes[next_node][0], self.position_nodes[next_node][1])))
                reward=self.model[next_node]

                new_reward = reward * 10 - np.log10(
                    distance.euclidean((self.position_nodes[current_node][0], self.position_nodes[current_node][1]),
                                       (self.position_nodes[next_node][0], self.position_nodes[next_node][1])))

                # print("Reward " + str(new_reward))
                self.update_q_values(current_node, next_node, new_reward)
                self.update_state_agent(next_node)
                self.update_model(next_node, reward)

            self.reset_agent()
    def reset_agent(self):

        loc_center = [distance.euclidean((self.L / 2, self.L / 2), i) for i in self.position_nodes]
        self.current_node = np.argmin(loc_center)
        self.t = 0

    def save_data(self):

        np.savetxt('./planning_learning_data/visited_states.csv', self.visite_states, delimiter=',')

        with open('./planning_learning_data/tabular.pkl', 'wb') as file:
            pickle.dump(self.table, file)


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
    model_data=True
    enviroment_data=False


    targets = np.loadtxt('target_large.csv', delimiter=',')
    target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]

    if not model_data:
        a1.deterministic_model(target_location)
        with open('./planning_learning_data/model.pkl', 'wb') as file:
            pickle.dump(a1.model, file)


    else:

        with open('./planning_learning_data/model.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.model = pickle.load(file)



    if not enviroment_data:
        a1.actual_model(target_location)
        with open('./planning_learning_data/model.pkl', 'wb') as file:
            pickle.dump(a1.enviroment, file)


    else:

        with open('./planning_learning_data/model.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.enviroment = pickle.load(file)



    #Ya tenemos modelo

    # for i in a1.nodes:
    #     for j in a1.nodes:
    #         try:
    #             print("Table at " +str((i,j)))
    #             a1.table[i,j]= -np.log10(np.log10(distance.euclidean(a1.position_nodes[i],a1.position_nodes[j])))
    #
    #         except:
    #
    #             a1.table[i, j]=-1000
    #Number of episodes

    for nn in range(100):

        targets = np.loadtxt('target_large.csv', delimiter=',')
        target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]
        # enviroment = Enviroment(L, target_location, radius_detection)

        print( "---------episode ---" + str(nn))

        episodic_enviroment=copy.deepcopy(a1.enviroment)

        for t in range(5000):

            current_node=a1.current_node
            next_node=a1.epsilon_greedy_action(current_node)
            # print("Going to " + str((a1.position_nodes[next_node][0],a1.position_nodes[next_node][1]) ))
            reward=episodic_enviroment[next_node]
            episodic_enviroment[next_node]=0
            new_reward=reward*10-np.log10(distance.euclidean((a1.position_nodes[current_node][0],a1.position_nodes[current_node][1]),
                                                 (a1.position_nodes[next_node][0],a1.position_nodes[next_node][1])))

            # print("Reward " + str(new_reward))
            a1.update_q_values(current_node, next_node, new_reward)
            a1.update_state_agent(next_node)
            a1.update_model(next_node, reward)

            time.sleep(2)

        print("targets leftovers " + str(np.sum(episodic_enviroment)))

        a1.reset_agent()
        a1.simulate_n_steps()



        # learning.append(500- len(enviroment.target_location))



    a1.save_data()



