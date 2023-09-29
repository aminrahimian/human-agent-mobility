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

    def __init__(self, T, load_weights, simulation):

        self.L = 10000
        self.pos_x = self.L / 2
        self.pos_y = self.L / 2
        self.alpha = 0.1
        if load_weights:

            self.weights = np.loadtxt('weights_SARSA.csv', delimiter=',')

        else:

            self.weights = np.array([0] + len(self.center_lay1) * [0] +
                                    [-1] + len(self.center_lay1) * [-1])

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
        self.width_cell = 25
        n_cells = int(np.floor(self.L / self.width_cell))

        self.dict_tabular = {0: np.eye(1),
                             1: np.eye(10),
                             2: np.eye(0)}





#####

