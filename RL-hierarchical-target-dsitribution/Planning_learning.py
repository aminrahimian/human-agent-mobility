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
        self.radius_detection = radius_detection
        self.a = np.sqrt(2)*self.radius_detection
        self.dim_table = (int(np.ceil(self.L/self.a)))
        self.t=0
        self.alpha=0.5
        self.epsilon=0.1/np.sqrt(self.t+1)
        self.n_planning=1000
        self.index_targets=[]
        index_0=int(np.floor(L*0.5/self.a))
        self.current_node=(index_0, index_0)
        self.pos_x = [self.a*0.5+ self.a*self.current_node[0]]
        self.pos_y = [self.a*0.5+ self.a*self.current_node[0]]
        self.history=[self.current_node]

        # be careful with this parameter


        self.model= np.zeros((self.dim_table, self.dim_table))
        self.enviroment=np.zeros((self.dim_table, self.dim_table))

        self.keys_adj = list(product(np.arange(self.dim_table),
                            np.arange(self.dim_table)))

        values_adj = [[]] * len(self.keys_adj)
        self.adjency=dict(zip(self.keys_adj, values_adj))
        self.q_values=dict(zip(self.keys_adj, values_adj))
        self.visite_states=[]

        # my_dicty[(1, 1, 1)] = (1, 2)

    def deterministic_model(self, target_location):

        new_target_location=[]

        for target in target_location:

            new_x=target[0]+norm.rvs(0,30,1)[0]
            new_y = target[1] + norm.rvs(0, 30, 1)[0]
            new_target_location.append((new_x,new_y))


        for i in new_target_location:
            coord_x = int(np.floor(i[0] / self.a))
            coord_y = int(np.floor(i[1] / self.a))

            self.model[coord_x, coord_y] += 1

            self.index_targets.append((coord_x, coord_y))


        return self.index_targets
    def actual_model(self, target_location):


        for i in target_location:

            coord_x = int(np.floor(i[0] / self.a))
            coord_y = int(np.floor(i[1] / self.a))

            self.enviroment[coord_x, coord_y] += 1

    def epsilon_greedy_action(self,index_x, index_y):

        for ngh in self.adjency[(index_x, index_y)]:

            if ngh in self.visite_states:

                index_ngh=self.adjency[(index_x, index_y)].index(ngh)
                self.q_values[(index_x, index_y)][index_ngh]=-1e6

        treshold=uniform.rvs(0,1)

        # print("===============================================")

        if treshold>=self.epsilon:


            index_max=np.argmax(self.q_values[(index_x, index_y)])
            # print("Index_max : " + str(index_max) + " q_val " + str(self.q_values[(index_x, index_y)][index_max]))
            # print("Origin : " + str((index_x, index_y)))
            next_node=self.adjency[(index_x, index_y)][index_max]
            # print("Adjency : " +str(self.adjency[(index_x, index_y)]))
            #
            # print("Q values : " +str(self.q_values[(index_x, index_y)]))
            # print("Next_node " + str(next_node))
            # if (index_x,index_y) in self.index_targets:


            return (next_node, index_max)

        else:

            index_choice=list(range(len(self.adjency[(index_x, index_y)])))
            index_max=np.random.choice(index_choice)
            # print("Index_max : " + str(index_max) + " q_val " + str(self.q_values[(index_x, index_y)][index_max]))
            # print("Origin : " + str((index_x, index_y)))
            next_node=self.adjency[(index_x, index_y)][index_max]
            # print("Adjency : " + str(self.adjency[(index_x, index_y)]))
            # print("Q values : " + str(self.q_values[(index_x, index_y)]))
            # # print("Random nodes : " +str(reduced_nods))
            #
            # print("Next_node " + str(next_node))
            # if (index_x, index_y) in self.index_targets:


            return (next_node, index_max)

    def update_state_agent(self, index_x_prime, index_y_prime):

        # print("Moving to :"  +str((new_pos_x, new_pos_y)))

        self.current_node=(index_x_prime, index_y_prime)
        self.pos_x.append((index_x_prime+0.5)*self.a)
        self.pos_y.append((index_y_prime+0.5)*self.a)
        self.t+=1

    def update_q_values(self,index_x, index_y, index_x_prime, index_y_prime, index_max, reward):

        # print("Previous value: " + str(self.q_values[(index_x, index_y)][index_max]))

        self.q_values[(index_x, index_y)][index_max]=\
            self.q_values[(index_x, index_y)][index_max]+\
            self.alpha*(reward+max(self.q_values[(index_x_prime, index_y_prime)])-
                        self.q_values[(index_x, index_y)][index_max] )


        try:

            # print(" Updating q value back ")
            index_back=self.adjency[(index_x_prime, index_y_prime)].index((index_x, index_y))
            self.q_values[(index_x_prime, index_y_prime)][index_back]=-1000

        except:

            pass

        # print("Updated value :" + str(self.q_values[(index_x, index_y)][index_max]))

        # print("Updated value :" + str(self.table[triplet_index[0],triplet_index[1], triplet_index[2]] ))

    def update_q_values_end(self, index_x, index_y, index_x_prime, index_y_prime, index_max, reward):

        print("Previous value: " +  str(self.q_values[(index_x, index_y)][index_max]))

        self.q_values[(index_x, index_y)][index_max] = \
            self.q_values[(index_x, index_y)][index_max] + \
            self.alpha * (reward -
                          self.q_values[(index_x, index_y)][index_max])


        try:

            # print(" Updating q value back ")
            index_back=self.adjency[(index_x_prime, index_y_prime)].index((index_x, index_y))
            self.q_values[(index_x_prime, index_y_prime)][index_back]=-1000

        except:

            pass


        print("Updated value :" + str(self.q_values[(index_x, index_y)][index_max]) )

    def update_model(self, index_x_prime,index_y_prime, reward):

        self.model[( index_x_prime,index_y_prime)]=reward

    def simulate_n_steps(self):

        self.epsilon=0.1
        planning_model = copy.deepcopy(self.model)

        for k in range(self.n_planning):

            index_x, index_y = self.current_node
            self.history.append((index_x, index_y))
            next_node, index_max = self.epsilon_greedy_action(index_x, index_y)

            index_x_prime = next_node[0]
            index_y_prime = next_node[1]
            # print("location " + str((index_x,index_y)))
            # print("Vecinos " + str(a1.adjency[(index_x_prime,index_y_prime)]))

            reward = planning_model[next_node]
            planning_model[next_node] = 0
            pos_x = self.a * (index_x + 0.5)
            pos_y = self.a * (index_y + 0.5)
            pos_x_prime = self.a * (index_x_prime + 0.5)
            pos_y_prime = self.a * (index_y_prime + 0.5)

            new_reward = reward * 10 - np.log2(distance.euclidean((pos_x, pos_y), (pos_x_prime, pos_y_prime)))

            # print("Reward " + str(new_reward))
            self.update_q_values(index_x, index_y, index_x_prime, index_y_prime, index_max, new_reward)

            self.update_state_agent(index_x_prime, index_y_prime)


        self.reset_agent()

    def reset_agent(self):

            index_0=int(np.floor(self.L*0.5/self.a))
            self.current_node = (index_0, index_0)
            self.t = 0
            self.epsilon=0.1/np.sqrt(self.t+1)
            self.pos_x = [self.a * 0.5 + self.a * self.current_node[0]]
            self.pos_y = [self.a * 0.5 + self.a * self.current_node[0]]


            self.history=[]

    def q_values_initialization(self, index_targets):

        amplied_list = list(np.arange(0, self.dim_table))
        amplied_list.append(amplied_list[-2])
        amplied_list = [1] + amplied_list

        conect_origen = []
        conect_dest = []

        for i in range(1, 10):
            conect_dest.append(index_targets[i * 50])
            conect_origen.append(index_targets[i * 50 - 1])

        # print("amplified list " + str(amplied_list[-10:-1])+ " last ele " + str(amplied_list[-1]))
        #
        # time.sleep(5)

        for i in range(1, self.dim_table + 1):

            for j in range(1, self.dim_table + 1):

                sub_list_i = amplied_list[i - 1:i + 2]
                sub_list_j = amplied_list[j - 1:j + 2]

                destiny = list(itertools.product(sub_list_i, sub_list_j))
                # print(" vecinos " + str(destiny))
                print("Current state " + str((amplied_list[i], amplied_list[j])))
                # print(" current esta en vecinos " + str((amplied_list[i], amplied_list[j]) in destiny)  )


                destiny.remove((amplied_list[i], amplied_list[j]))


                if (amplied_list[i], amplied_list[j]) in conect_origen:

                    print(" Connection ****************************")
                    time.sleep(2)
                    index_look=conect_origen.index((amplied_list[i], amplied_list[j]))
                    destiny.append(conect_dest[index_look])


                self.adjency[(amplied_list[i], amplied_list[j])] = destiny
                # print("for pair " + str((amplied_list[i], amplied_list[j])) + "neogh " + str(destiny))

                self.q_values[(amplied_list[i], amplied_list[j])] = [-np.log10(
                    distance.euclidean(((amplied_list[i]) * self.a + self.a * 0.5, (amplied_list[j]) * self.a + self.a * 0.5),
                                       ((k[0]) * self.a + self.a * 0.5, (k[1]) * self.a + self.a * 0.5))) +
                                               10 * self.model[(amplied_list[i], amplied_list[j])] for k in destiny]

                # time.sleep(2)


                if (amplied_list[i], amplied_list[j]) in self.adjency[(amplied_list[i], amplied_list[j])]:

                    print(" SELF LOOP  ************************************************* ")
                    # time.sleep(2)

                    index_delete = self.adjency[(amplied_list[i], amplied_list[j])].index((amplied_list[i], amplied_list[j]))
                    self.adjency[(amplied_list[i], amplied_list[j])].remove((amplied_list[i], amplied_list[j]))
                    self.q_values[(amplied_list[i], amplied_list[j])].pop(index_delete)



        return self.adjency

    def save_data(self):

        np.savetxt('./planning_learning_data/visited_states.csv', self.visite_states, delimiter=',')

        with open('../planning_learning_data/q_values.pkl.pkl', 'wb') as file:
            pickle.dump(self.q_values, file)


        with open('../planning_learning_data/model.pkl', 'wb') as file:
            pickle.dump(self.model, file)

        # with open('./planning_learning_data/sample_dict.pkl', 'wb') as file:
        #     pickle.dump(self.sample_dict, file)

    def load_data(self):

        self.visite_states=list(np.loadtxt('./planning_learning_data/visited_states.csv', delimiter=','))

        with open('../planning_learning_data/table.pkl', 'rb') as file:
            # Call load method to deserialze
            self.table = pickle.load(file)

        with open('../planning_learning_data/model.pkl', 'rb') as file:
            # Call load method to deserialze
            self.model = pickle.load(file)

        with open('../planning_learning_data/sample_dict.pkl', 'rb') as file:
            # Call load method to deserialze
            self.sample_dict = pickle.load(file)

if __name__ == "__main__":

    L=10000
    T=20
    radius_detection=25
    a1=Agent(L,T, radius_detection)
    model_data=True
    enviroment_data=True
    q_values_data=False


    targets = np.loadtxt('target_large.csv', delimiter=',')
    target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]

    if not model_data:

        index_targets=a1.deterministic_model(target_location)

        with open('../planning_learning_data/model.pkl', 'wb') as file:
            pickle.dump(a1.model, file)

        with open('../planning_learning_data/index_targets.pkl', 'wb') as file:
            pickle.dump(index_targets, file)

    else:

        with open('../planning_learning_data/model.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.model = pickle.load(file)

        with open('../planning_learning_data/index_targets.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.index_targets = pickle.load(file)

    if not enviroment_data:
        a1.actual_model(target_location)
        with open('../planning_learning_data/enviroment.pkl', 'wb') as file:
            pickle.dump(a1.enviroment, file)

    else:

        with open('../planning_learning_data/enviroment.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.enviroment = pickle.load(file)


    index_targets=a1.index_targets

    if not q_values_data:

        a1.adjency=a1.q_values_initialization(index_targets)

        with open('../planning_learning_data/q_values.pkl', 'wb') as file:
            pickle.dump(a1.q_values, file)

        with open('../planning_learning_data/adjency_matrix.pkl', 'wb') as file:
            pickle.dump(a1.adjency, file)

        # np.savetxt('./planning_learning_data/table.out', a1.table, delimiter=',')

    else:

        with open('../planning_learning_data/q_values.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.q_values=pickle.load(file)

        with open('../planning_learning_data/adjency_matrix.pkl', 'rb') as file:
            # Call load method to deserialze
            a1.adjency = pickle.load(file)

    learning_rate=[]

    for i in a1.q_values.keys():

        if any(np.isinf(a1.q_values[i])):

            print(" inf vals in :" + str(a1.adjency[i])+ " ")

    for nn in range(5000):

        #add cumulative distance and rewards to calculate search efficiency
        #
        # a1.epsilon=0.5/np.log(nn+1)
        a1.alpha=0.5
        targets = np.loadtxt('target_large.csv', delimiter=',')
        target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]
        # enviroment = Enviroment(L, target_location, radius_detection)

        print( "---------episode ---" + str(nn))

        episodic_enviroment=copy.deepcopy(a1.enviroment)

        for t in range(5000-1):



            # print("step " +str(t))
            index_x,index_y=a1.current_node
            a1.history.append((index_x, index_y))
            next_node, index_max=a1.epsilon_greedy_action(index_x, index_y)

            index_x_prime=next_node[0]
            index_y_prime=next_node[1]
            # print("location " + str((index_x,index_y)))
            # print("Vecinos " + str(a1.adjency[(index_x_prime,index_y_prime)]))

            reward=episodic_enviroment[next_node]
            a1.model[next_node]=reward
            episodic_enviroment[next_node]=0
            pos_x=a1.a*(index_x+0.5)
            pos_y = a1.a*(index_y + 0.5)
            pos_x_prime = a1.a*(index_x_prime + 0.5)
            pos_y_prime = a1.a*(index_y_prime + 0.5)


            new_reward=reward*10-np.log2(distance.euclidean((pos_x, pos_y),(pos_x_prime, pos_y_prime)))

            if reward > 0:

                a1.q_values[(index_x, index_y)].append(new_reward)

                a1.index_targets.append((index_x, index_y))
                a1.adjency[(index_x, index_y)].append((index_x_prime,index_y_prime))


            # print("Reward " + str(new_reward))
            a1.update_q_values(index_x, index_y, index_x_prime, index_y_prime,index_max,new_reward)

            a1.update_state_agent(index_x_prime, index_y_prime)
            a1.update_model(index_x_prime, index_y_prime, reward)

            # time.sleep(2)




        index_x, index_y = a1.current_node
        a1.history.append((index_x, index_y))
        next_node, index_max = a1.epsilon_greedy_action(index_x, index_y)
        index_x_prime, index_y_prime = next_node
        # print("Going to " + str((a1.position_nodes[next_node][0],a1.position_nodes[next_node][1]) ))
        reward = episodic_enviroment[next_node]
        episodic_enviroment[next_node] = 0
        pos_x = a1.a * (index_x + 0.5)
        pos_y = a1.a * (index_y + 0.5)
        pos_x_prime = a1.a * (index_x_prime + 0.5)
        pos_y_prime = a1.a * (index_y_prime + 0.5)

        if reward > 0:
            a1.q_values[(index_x, index_y)].append(new_reward)
            a1.index_targets.append((index_x, index_y))
            a1.adjency[(index_x, index_y)].append((index_x_prime, index_y_prime))

        new_reward = reward * 10 - np.log2(distance.euclidean((pos_x, pos_y), (pos_x_prime, pos_y_prime)))

        # print("Reward " + str(new_reward))
        a1.update_q_values_end(index_x, index_y, index_x_prime, index_y_prime,index_max,new_reward)
        a1.update_state_agent(index_x_prime, index_y_prime)
        a1.update_model(index_x_prime, index_y_prime, reward)


        # time.sleep(2)

        print("targets leftovers " + str(np.sum(episodic_enviroment)))

        numerator=500-np.sum(episodic_enviroment)
        denominator=np.sum([distance.euclidean((a1.pos_x[i], a1.pos_y[i]),(a1.pos_x[i+1], a1.pos_y[i+1])) for i in range(4999)])
        learning_rate.append(numerator/denominator)

        scientific_notation = "{:e}".format(learning_rate[-1])
        print("Search efficiency " +scientific_notation)
        a1.reset_agent()
        a1.simulate_n_steps()


        # learning.append(500- len(enviroment.target_location))


    a1.save_data()

    with open('../planning_learning_data/learning_rate_2000.pkl', 'wb') as file:
        pickle.dump(learning_rate, file)

    # matrix of rewards

    xpoints = list(range(5000))

    new_val=learning_rate[1]
    learning_rate[0] =new_val
    ypoints = learning_rate

    new_lr=[float('{:f}'.format(a)) for a in learning_rate]
    type(new_lr[0])

    plt.ticklabel_format(axis='y', style='sci')
    plt.xlabel('Episode ')
    plt.ylabel('Search efficiency')
    plt.plot(xpoints, new_lr)
    plt.show()

    plt.ticklabel_format(axis="x", style="sci", )
    plt.show()

    #
    # a1.adjency[(159,64)]

    with open('../planning_learning_data/q_values.pkl.pkl', 'wb') as file:
            pickle.dump(a1.q_values, file)


    with open('../planning_learning_data/model.pkl', 'wb') as file:
        pickle.dump(a1.model, file)



##
        #
def generate_path():

        a1.alpha=0.2
        targets = np.loadtxt('target_large.csv', delimiter=',')
        target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]
        # enviroment = Enviroment(L, target_location, radius_detection)

        print( "---------episode ---" + str(nn))

        episodic_enviroment=copy.deepcopy(a1.enviroment)

        for t in range(5000-1):


            # print("step " +str(t))
            index_x,index_y=a1.current_node
            a1.history.append((index_x, index_y))
            next_node, index_max=a1.epsilon_greedy_action(index_x, index_y)

            index_x_prime=next_node[0]
            index_y_prime=next_node[1]
            # print("location " + str((index_x,index_y)))
            # print("Vecinos " + str(a1.adjency[(index_x_prime,index_y_prime)]))

            reward=episodic_enviroment[next_node]
            a1.model[next_node]=reward
            episodic_enviroment[next_node]=0
            pos_x=a1.a*(index_x+0.5)
            pos_y = a1.a*(index_y + 0.5)
            pos_x_prime = a1.a*(index_x_prime + 0.5)
            pos_y_prime = a1.a*(index_y_prime + 0.5)


            new_reward=reward*10-np.log10(distance.euclidean((pos_x, pos_y),(pos_x_prime, pos_y_prime)))

            if reward > 0:

                a1.q_values[(index_x, index_y)].append(new_reward)
                a1.index_targets.append((index_x, index_y))
                a1.adjency[(index_x, index_y)].append((index_x_prime,index_y_prime))


            # print("Reward " + str(new_reward))
            a1.update_q_values(index_x, index_y, index_x_prime, index_y_prime,index_max,new_reward)

            a1.update_state_agent(index_x_prime, index_y_prime)
            a1.update_model(index_x_prime, index_y_prime, reward)




            # time.sleep(2)
        index_x, index_y = a1.current_node
        a1.history.append((index_x, index_y))
        next_node, index_max = a1.epsilon_greedy_action(index_x, index_y)
        index_x_prime, index_y_prime = next_node
        # print("Going to " + str((a1.position_nodes[next_node][0],a1.position_nodes[next_node][1]) ))
        reward = episodic_enviroment[next_node]
        episodic_enviroment[next_node] = 0
        pos_x = a1.a * (index_x + 0.5)
        pos_y = a1.a * (index_y + 0.5)
        pos_x_prime = a1.a * (index_x_prime + 0.5)
        pos_y_prime = a1.a * (index_y_prime + 0.5)

        if reward > 0:
            a1.q_values[(index_x, index_y)].append(new_reward)
            a1.index_targets.append((index_x, index_y))
            a1.adjency[(index_x, index_y)].append((index_x_prime, index_y_prime))

        new_reward = reward * 10 - np.log10(distance.euclidean((pos_x, pos_y), (pos_x_prime, pos_y_prime)))

        # print("Reward " + str(new_reward))
        a1.update_q_values_end(index_x, index_y, index_x_prime, index_y_prime,index_max,new_reward)
        a1.update_state_agent(index_x_prime, index_y_prime)
        a1.update_model(index_x_prime, index_y_prime, reward)


        t_x=[target[0] for target in target_location]
        t_y=[target[1] for target in target_location]

        colors=list(range(2000))
        # color = '#94F4EE'
        colors= [ 0.0005*i for i in colors]

        plt.plot(a1.pos_x, a1.pos_y, linestyle='dashed', linewidth=0.5, color = 'b')
        plt.scatter(t_x, t_y, s=4, c='#F04C1C')
        # plt.text(a1.pos_x[0] - 0.015, a1.pos_y[0] + 0.25, "Step 0")
        # plt.text(a1.pos_x[500] - 0.015, a1.pos_y[500] + 0.25, "Step 500")
        # plt.text(a1.pos_x[750] - 0.015, a1.pos_y[750] + 0.25, "Step 750")
        # plt.text(a1.pos_x[2000] - 0.015, a1.pos_y[2000] + 0.25, "Step 2000")
        # plt.text(a1.pos_x[3000] - 0.015, a1.pos_y[3000] + 0.25, "Step 3000")
        # plt.text(a1.pos_x[3500] - 0.015, a1.pos_y[3500] + 0.25, "Step 3500")
        # plt.text(a1.pos_x[4000] + 0.215, a1.pos_y[4000] - 0.015, "Step 4000")
        # plt.text(a1.pos_x[5000] - 0.015, a1.pos_y[5000] + 0.25, "Step 5000")

        plt.plot
        plt.show()




  with open('../planning_learning_data/q_values.pkl', 'wb') as file:
            pickle.dump(a1.q_values, file)



  with open('../planning_learning_data/pos_x.pkl', 'wb') as file:
            pickle.dump(a1.pos_x, file)

  with open('../planning_learning_data/pos_y.pkl', 'wb') as file:
            pickle.dump(a1.pos_y, file)

# check for inf values


conect_origen=[]
conect_dest=[]

for i in range(1,10):

    conect_dest.append(index_targets[i*50])
    conect_origen.append(index_targets[i * 50-1])




kernel_lr=[]

for i in range(100,4900):

    kernel_lr.append(np.mean(learning_rate[i-100:i+100]))



xpoints=list(range(len(kernel_lr)))
plt.ticklabel_format(axis='y', style='sci')
plt.xlabel('Episode ')
plt.ylabel('Search efficiency')
plt.plot(xpoints, kernel_lr, color = 'b')
plt.axhline(y = 2.2e-4, color = 'r', linestyle = '-')

plt.show()

plt.ticklabel_format(axis="x", style="sci", )
plt.show()

    #

