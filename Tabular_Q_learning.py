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

class Enviroment:

    def __init__(self, targets_coord):

        self.L=10000
        self.targets_coord=targets_coord

    def plot_target(self):

        x_t = [self.targets_coord[i][0] for i in range(len(self.targets_coord))]
        y_t = [self.targets_coord[i][1] for i in range(len(self.targets_coord))]

        plt.plot(x_t, y_t, "ok", label="input point", color="red")
        # plt.legend()
        # plt.colorbar()
        plt.xlim((0,10000))
        plt.ylim((0,10000))
        # plt.axis("equal")
        plt.show()

    def collected_targets(self, pos_x, pos_y, radius_detection):

        """ Return the number of targets given the historical
        position of the agent
        """
        agent_position = (pos_x, pos_y)

        cont = 0
        initial_range = len(self.targets_coord)
        removal = set({})
        list_target_location = list(self.targets_coord)

        for i in range(initial_range):

            if distance.euclidean(agent_position, list_target_location[i]) <= radius_detection:

                removal.add(list_target_location[i])
                cont += 1

            else:

                pass

        self.targets_coord = self.targets_coord.difference(removal)

        if cont > 0:

            R=100*cont
            print("Reward  : " + str(R))

        else:

            R=-10
            # print("No targets  -> Reward: " + str(R))


        return R

class Agent:

    def __init__(self, T, load_table, simulation):

        self.L = 10000
        self.pos_x = self.L / 2
        self.pos_y = self.L / 2
        self.alpha = 0.1
        self.width_cell = 25
        n_cells = int(np.floor(self.L / self.width_cell))

        if load_table:

            self.dict_tabular  = np.loadtxt('weights_SARSA.csv', delimiter=',')

        else:

            self.dict_tabular = {0: np.zeros((1,2)),
                                 1: np.zeros((n_cells, n_cells, 2)),
                                 2: np.zeros((n_cells, n_cells, 2))}

        # when simulation
        if simulation:

            self.t = 0

        else:

            self.t = 10
        # when ploting

        self.T = T
        self.path_x = []
        self.path_y = []
        self.epsilon = 0.1
        # define these parameters carefully
        self.radius_detection = 25
        self.terminal_state = False

    def q_values(self, pos_x, pos_y,time_step):

        index_x = int(np.floor(pos_x / self.width_cell))
        index_y = int(np.floor(pos_y / self.width_cell))

        print("indices "  +str((index_x, index_y)))

        if not time_step==self.T:

            if time_step==0:

                return self.dict_tabular[0]

            else:

                return self.dict_tabular[1][index_x, index_y,:]

        else:

            return self.dict_tabular[2][index_x, index_y,:]

    def sample_step_length(self, mu):

        alpha = mu - 1
        V = np.random.uniform(-np.pi * 0.5, 0.5 * np.pi, 1)[0]
        W = expon.rvs(size=1)[0]
        cop1 = (np.sin(alpha * V)) / ((np.cos(V)) ** (1 / alpha))
        cop2 = ((np.cos((1 - alpha) * V)) / W) ** (((1 - alpha) / alpha))
        l = cop1 * cop2

        ## set the

        if l <= 0:

            l = 10 * min(500, -l)
            step_l = max(25, l)

        else:

            l = 10 * min(500, l)
            step_l = max(25, l)

        # print("Taking this value:  " + str(elle))

        # angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # # print("angle vel :" + str(angle))
        # vel_x = 10 * elle * np.cos(angle)
        # vel_y = 10 * elle * np.sin(angle)

        return step_l

    def epsilon_greedy_policy(self, pos_x, pos_y):

        """
        Return an action [a] considering the q(s,a) chosing max{q(s,a)}
        with probability 1-epsilon
        :param pos_x: position x
        :param pos_y: position y
        :return: parameter mu of levy distribution
        """

        index_x = int(np.floor(pos_x / self.width_cell))
        index_y = int(np.floor(pos_y / self.width_cell))

        print("indices " + str((index_x, index_y)))

        if not self.terminal_state:

            if self.t == 0:

                q_values= self.dict_tabular[0]

            else:

                q_values= self.dict_tabular[1][index_x, index_y, :]

        else:

            q_values=self.dict_tabular[2][index_x, index_y, :]

        print("q values :" +str(q_values))

        sample_uniform=np.random.uniform(0, 1,1)[0]

        if sample_uniform>=self.epsilon:

            # print("Action taking by MAX  ::::   ::::  ")

            action=np.argmax(q_values)

        else:

            # print("Action taking by SAMPLING  ::::   ::::  ")
            action=np.random.choice(np.array([0,1]))

        return int(action)

    def next_position(self, action, pos_x, pos_y):

        if action==0:

            mu=2

        else:

            mu=3

        lenght_step = self.sample_step_length(mu)
        # print(" length step : " + str(lenght_step))
        angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # print("angle vel :" + str(angle))
        vel_x = lenght_step * np.cos(angle)
        vel_y = lenght_step * np.sin(angle)

        pos_x_prime = pos_x + vel_x
        pos_y_prime = pos_y + vel_y

        if pos_x_prime > self.L:
            pos_x_prime = 2 * self.L - pos_x_prime

        if pos_x_prime < 0:
            self.pos_x_prime = -1 * pos_x_prime

        if pos_y_prime > self.L:
            pos_y_prime = 2 * self.L - pos_y_prime

        if pos_y_prime < 0:
            pos_y_prime = -1 * pos_y_prime

        return (pos_x_prime, pos_y_prime)

    def update_agent_state(self, pos_x_prime, pos_y_prime):

        self.path_x.append(self.pos_x)
        self.path_y.append(self.pos_y)
        self.pos_x = pos_x_prime
        self.pos_y = pos_y_prime
        self.t += 1

        ## check this one

        if self.t == self.T:
            self.terminal_state = True

    def reset_agent(self):

        self.pos_x = self.L / 2
        self.pos_y = self.L / 2
        self.t = 0
        self.path_x = []
        self.path_y = []
        self.terminal_state = False

    def plot_path(self, targets_coord):

        targets_x = []
        targets_y = []
        targets = list(targets_coord)

        for r in targets:
            targets_x.append(r[0])
            targets_y.append(r[1])

        plt.plot(self.path_x, self.path_y, '--o', color='#65C7F5')
        plt.plot(targets_x, targets_y, 'ko')
        plt.xlim([0, 10000])
        plt.ylim([0, 10000])
        plt.show()

    def update_tabular_q_values(self, pos_x, pos_y, pos_x_prime, pos_y_prime, a,
                                reward):

        index_x = int(np.floor(pos_x / self.width_cell))
        index_y = int(np.floor(pos_y / self.width_cell))
        time_step=self.t

        if not time_step==self.T:

            if self.t == 0:

                print("for state and action " + str((index_x, index_y, a)) + " prev q value: " + str(self.dict_tabular[0][0][a]))

                self.dict_tabular[0][0][a]=self.dict_tabular[0][0][a]+self.alpha*(reward+max(self.q_values(pos_x_prime, pos_y_prime,(time_step+1))) -self.dict_tabular[0][0][a])

                print("for state and action " + str((index_x, index_y, a)) + " updated q value: " + str(
                    self.dict_tabular[0][0][a]))

            else:

                print("for state and action " + str((index_x, index_y, a)) + " prev q value: " + str(
                    self.dict_tabular[1][index_x, index_y,a]))

                self.dict_tabular[1][index_x, index_y,a]=self.dict_tabular[1][index_x, index_y,a]+ \
                                                         self.alpha * (reward + max(self.q_values(pos_x_prime, pos_y_prime,(time_step+1))) -
                                                                       self.dict_tabular[1][index_x, index_y,a])

                print("for state and action " + str((index_x, index_y, a)) + " updated q value: " + str(
                    self.dict_tabular[1][index_x, index_y, a]))


        else:

            print("for state and action " + str((index_x, index_y, a)) + " prev q value: " + str(
                self.dict_tabular[2][index_x, index_y, a]))

            self.dict_tabular[2][index_x, index_y, a] = self.dict_tabular[2][index_x, index_y, a] + \
                                                        self.alpha * (reward  -self.dict_tabular[2][index_x, index_y, a])

            print("for state and action " + str((index_x, index_y, a)) + " updated q value: " + str(
                self.dict_tabular[2][index_x, index_y, a]))





load_table=False
simulation=True
agent=Agent(1000,load_table, simulation)

np.argmax(agent.dict_tabular[0])

# agent.q_value(500, 500, 1)

for n in range(10):

    pos_x = agent.pos_x
    pos_y = agent.pos_y
    print("pos agent  " + str((pos_x, pos_y)) + "time : "  +str(agent.t))
    action = agent.epsilon_greedy_policy( pos_x, pos_y)
    print("action  " + str(action))
    pos_x_prime, pos_y_prime=agent.next_position(action, pos_x, pos_y)
    reward=-10
    agent.update_tabular_q_values(pos_x,pos_y,pos_x_prime,pos_y_prime,action,reward)
    time.sleep(4)
    agent.update_agent_state(pos_x_prime, pos_y_prime)
    print("-----------------------------------------------")



np.sum(agent.dict_tabular[1][:,:,0]<0)
