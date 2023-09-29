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


    def q_value(self, pos_x, pos_y, action):

        index_x = int(np.floor(pos_x / self.width_cell))
        index_y = int(np.floor(pos_y / self.width_cell))

        if not self.terminal_state:

            if self.t==0:

                return self.dict_tabular[0][action]

            else:

                return self.dict_tabular[1][index_x, index_y, action]

        else:

            return 0




#####


load_table=False
simulation=True
agent=Agent(1000,load_table, simulation)

agent.dict_tabular[1][0,1,1]=10

