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

    def reset_enviroment(self):

        self.L = L
        self.target_location = target_location
        self.cont_prev = 0
