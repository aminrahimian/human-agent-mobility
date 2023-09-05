import numpy as np
from scipy.stats import norm
import itertools
from scipy.spatial import distance
import copy
import pickle
import matplotlib.pyplot as plt
import time


# generate targets.
class Enviroment:
    def __init__(self, L, n_cluster,corr,n_target_1, n_target_2=0,n_target_3=0):

        self.L = L
        target_global_sd_x=10
        target_global_sd_y=10
        target_cluster_sd_x = 4
        target_cluster_sd_y = 4
        self.n_threats=5

        target_global_mean_x = 0.5 * L
        target_global_mean_y = 0.5 * L

        if n_cluster==1:

            center_target_1_x = norm.rvs(loc=target_global_mean_x, scale=target_global_sd_x, size=1)[0]
            center_target_1_y = norm.rvs(loc=target_global_mean_y, scale=target_global_sd_y, size=1)[0]

            target_1_x = norm.rvs(loc=center_target_1_x, scale=target_cluster_sd_x, size=n_target_1)
            target_1_y = norm.rvs(loc=center_target_1_y, scale=target_cluster_sd_y, size=n_target_1)

            target_2_x = norm.rvs(loc=0, scale=0, size=0)
            target_2_y = norm.rvs(loc=0, scale=0, size=0)

            target_3_x = norm.rvs(loc=0, scale=0, size=0)
            target_3_y = norm.rvs(loc=0, scale=0, size=0)
            self.n_targets = n_target_1 + n_target_2 + n_target_3

            target_x = np.concatenate((target_1_x, target_2_x, target_3_x))
            target_y = np.concatenate((target_1_y, target_2_y, target_3_y))

        elif n_cluster==2:

            center_target_1_x = norm.rvs(loc=target_global_mean_x, scale=target_global_sd_x, size=n_cluster)
            center_target_1_y = norm.rvs(loc=target_global_mean_y, scale=target_global_sd_y, size=n_cluster)

            target_1_x = norm.rvs(loc=center_target_1_x[0], scale=target_cluster_sd_x, size=n_target_1)
            target_1_y = norm.rvs(loc=center_target_1_y[0], scale=target_cluster_sd_y, size=n_target_1)

            target_2_x = norm.rvs(loc=center_target_1_x[1], scale=target_cluster_sd_x, size=n_target_2)
            target_2_y = norm.rvs(loc=center_target_1_y[1], scale=target_cluster_sd_y, size=n_target_2)

            target_3_x = norm.rvs(loc=0, scale=0, size=0)
            target_3_y = norm.rvs(loc=0, scale=0, size=0)

            self.n_targets = n_target_1 + n_target_2 + n_target_3

            target_x = np.concatenate((target_1_x, target_2_x, target_3_x))
            target_y = np.concatenate((target_1_y, target_2_y, target_3_y))

        else:

            center_target_1_x = norm.rvs(loc=target_global_mean_x, scale=target_global_sd_x, size=n_cluster)
            center_target_1_y = norm.rvs(loc=target_global_mean_y, scale=target_global_sd_y, size=n_cluster)

            target_1_x = norm.rvs(loc=center_target_1_x[0], scale=target_cluster_sd_x, size=n_target_1)
            target_1_y = norm.rvs(loc=center_target_1_y[0], scale=target_cluster_sd_y, size=n_target_1)

            target_2_x = norm.rvs(loc=center_target_1_x[1], scale=target_cluster_sd_x, size=n_target_2)
            target_2_y = norm.rvs(loc=center_target_1_y[1], scale=target_cluster_sd_y, size=n_target_2)

            target_3_x = norm.rvs(loc=center_target_1_x[2], scale=target_cluster_sd_x, size=n_target_3)
            target_3_y = norm.rvs(loc=center_target_1_y[2], scale=target_cluster_sd_y, size=n_target_3)

            self.n_targets = n_target_1 + n_target_2 + n_target_3

            target_x = np.concatenate((target_1_x, target_2_x, target_3_x))
            target_y = np.concatenate((target_1_y, target_2_y, target_3_y))

        keys_dict_targets=list(range(self.n_targets))
        coordinates_targets = [(target_x[i], target_y[i],0) for i in range(self.n_targets)]

        # the first element is s, second a
        self.targets = dict(zip(keys_dict_targets, coordinates_targets))

        mean = np.array([target_global_mean_x, target_global_mean_x])
        sigmas = np.array([[target_global_sd_x, 0], [0, target_global_sd_x]])
        pearson_corr= np.array([[1, corr], [corr, 1]])

        cov=sigmas@pearson_corr@sigmas

        _, x_threats = np.random.multivariate_normal(mean, cov, self.n_threats).T
        _, y_threats = np.random.multivariate_normal(mean, cov, self.n_threats).T

        keys_dict_threats = list(range(self.n_threats))
        coordinates_threats = [(x_threats[i], y_threats[i], 0) for i in range(self.n_threats)]

        self.threats= dict(zip(keys_dict_threats, coordinates_threats))

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


class Agent:

    def __init__(self, radius_coarse, L, T,radius_detection, freq_sampling,episodes):

        self.L = L
        self.T = T
        self.pos_x = L/2
        self.pos_y = L/2
        self.pos_x_prime=L/2
        self.pos_y_prime=L/2
        grid_line = 1000 * np.arange(0.5, 10.5, 1)
        center_lay1 = list(itertools.product(grid_line, grid_line))
        center_lay2 = list(itertools.product(grid_line, grid_line))
        center_init = [(L/2, L/2)]
        self.centers = center_init + center_lay1 + center_lay2
        self.weigths_vector = [100] + len(center_lay1) * [100] + len(center_lay1) * [0]
        self.radius_coarse = radius_coarse
        self.radius_detection=radius_detection
        self.freq_sampling=freq_sampling
        self.cum_targets = 0
        self.alpha = 0.9
        self.gamma = 1
        self.time_step = 0
        self.episodes=episodes

    def feature_vector(self, pos_x, pos_y):

        actual_pos = len(self.centers) * [(pos_x,pos_y)]
        iter = len(self.centers)
        distances = np.array([distance.euclidean(self.centers[i], actual_pos[i]) for i in range(iter)])
        inside_circle = (distances <= self.radius_coarse)
        inside_circle = inside_circle.astype(int)

        return inside_circle

    def update_next_state(self,action):

        vel_x=action[0]
        vel_y=action[1]
        self.pos_x_prime = self.pos_x + vel_x * 1
        self.pos_y_prime = self.pos_y + vel_y * 1

    def update_state(self):

        self.pos_x = self.pos_x_prime
        self.pos_y = self.pos_y_prime

    def delta_value(self, reward, action):

        self.update_next_state(action)
        # print(a1.pos_x)
        # print(a1.pos_y)
        # print(a1.pos_x_prime)
        # print(a1.pos_y_prime)
        current_s = np.dot(self.feature_vector(self.pos_x, self.pos_y), self.weigths_vector)
        next_s = self.gamma * np.dot(self.feature_vector(self.pos_x_prime, self.pos_y_prime), self.weigths_vector)
        self.update_state()

        return reward +next_s-current_s


    def update_weigths(self):

        pass



    def


radius_coarse=750
L=10000
T=50
radius_detection=25
freq_sampling=1
episodes=50

a1=Agent(radius_coarse,L,T,radius_detection, freq_sampling, episodes)
print(a1.delta_value(10,(4,5)))






















