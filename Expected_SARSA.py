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


def generate_targets():
    """
    Generate targets around points x_i,y_i
    :return: list with targets coordinates as [(x1,y1), .., (xn,yn)]
    """

    x1 = 5000
    y1 = 7500
    x2 = 2500
    y2 = 7500
    x3 = 2500
    y3 = 2500
    rt = 250

    angles = np.linspace(0, 2 * np.pi, num=49, endpoint=False)
    p1 = [(x1 + rt * np.cos(i), y1 + rt * np.sin(i)) for i in angles]
    p1 = [(x1, y1)] + p1

    p2 = [(x2 + rt * np.cos(i), y2 + rt * np.sin(i)) for i in angles]
    p2 = [(x2, y2)] + p2

    p3 = [(x3 + rt * np.cos(i), y3 + rt * np.sin(i)) for i in angles]
    p3 = [(x3, y3)] + p3
    return p1 + p2 + p3

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

class Agent:

    def __init__(self):

        self.pos_x=0
        self.pos_y=0
        L=10000
        grid_line = 1000 * np.arange(0, 12, 2)
        self.center_lay1 = list(itertools.product(grid_line, grid_line))
        self.center_lay2 = list(itertools.product(grid_line, grid_line))
        self.center_init = [(L/2, L/2)]
        self.centers = self.center_init + self.center_lay1 + self.center_lay2
        self.alpha=1e-4
        self.weights= np.array([0] + len(self.center_lay1) * [0] + len(self.center_lay2) * [0])


    def feature_vector(self, pos_x, pos_y):

        pass

    def sample_step_length(self):

        self.A.append(self.sample_parameter())
        alpha = self.parameter - 1
        V = np.random.uniform(-np.pi * 0.5, 0.5 * np.pi, 1)[0]
        W = expon.rvs(size=1)[0]
        cop1 = (np.sin(alpha * V)) / ((np.cos(V)) ** (1 / alpha))
        cop2 = ((np.cos((1 - alpha) * V)) / W) ** (((1 - alpha) / alpha))
        elle = cop1 * cop2

        if elle <= 0:

            elle = min(500, -elle)

        else:

            elle = min(500, elle)

        # print("Taking this value:  " + str(elle))

        angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # print("angle vel :" + str(angle))
        vel_x = 10 * elle * np.cos(angle)
        vel_y = 10 * elle * np.sin(angle)

        return (vel_x, vel_y)






targets_coord=generate_targets()


e1=Enviroment(targets_coord)
e1.plot_target()

