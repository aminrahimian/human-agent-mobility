import math
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


class Enviroment:

    def __init__(self,target_location):

        self.targets=np.zeros((200,200))

        for i in target_location:

            x_index=int(np.floor(i[0]/50))
            y_index=int(np.floor(i[1]/50))

            self.targets[x_index,y_index]+=1


class Agent:

    def __init__(self, target_grid):

        self.pos_x=5000
        self.pos_y=5000
        self.L=10000
        self.w= np.zeros((20,20,10))
        self.theta_u_r=32*np.zeros((20,20,10))
        self.theta_sigma_r=0.35*np.zeros((20,20,10))
        self.theta_u_angle=np.zeros((20,20,10))
        self.theta_sigma_angle=0.12*np.zeros((20,20,10))
        self.alpha_w=1e-2
        self.alpha_u_r=1e-5
        self.alpha_sigma_r=1e-8
        self.alpha_u_angle=3e-5
        self.alpha_sigma_angle=1e-6
        self.collected_targets=0
        self.distance_traveled=0
        self.path_x=[self.pos_x]
        self.path_y=[self.pos_y]
        self.updates_theta_u_r=[]
        self.updates_theta_sigma_r=[]
        self.updates_theta_u_angle=[]
        self.updates_theta_sigma_angle=[]
        self.target_grid=target_grid


    def pos_to_feature(self, pos_x, pos_y):

        feature_vector = np.zeros((20, 20, 10))

        pairs = [(int(np.floor((pos_x - i * 50) / 500)), int(np.floor((pos_y - i * 50) / 500))) for i in
                 range(10)]

        index = 0
        pair = pairs[index]

        while (pair[0] >= 0 and pair[1] >= 0):

            feature_vector[pair[0], pair[1], index] = 1
            index += 1
            # print("index " + str(index))
            if index > 9:
                break

            pair = pairs[index]

        return feature_vector



    def initialize_thetas(self):

        x_samples = uniform.rvs(0, 10000, size=8000)
        y_samples = uniform.rvs(0, 10000, size=8000)

        for row in range(8000):

            pos_x = x_samples[row]
            pos_y = y_samples[row]

            mask_matrix = self.pos_to_feature(pos_x, pos_y).astype(bool)

            #
            # if (pos_x>=0 ) and (pos_x<=3000) and (pos_y>= 5000) and (pos_y<= 10000):
            #
            #     set_angle=np.pi*0.5
            #
            # elif (pos_x >= 0) and (pos_x <= 3000) and (pos_y >= 0) and (pos_y <= 5000):
            #
            #     set_angle = -np.pi * 0.5
            #
            # elif (pos_x >= 3000) and (pos_x <= 7000) and (pos_y >= 6000) and (pos_y <= 10000):
            #     set_angle = 0
            #
            # elif (pos_x >= 3000) and (pos_x <= 10000) and (pos_y >= 4000) and (pos_y <= 6000):
            #
            #     set_angle = np.pi
            #     print("centro")
            #
            #
            # elif (pos_x >= 3000) and (pos_x <= 7000) and (pos_y >= 0) and (pos_y <= 4000):
            #
            #     set_angle = 0
            #
            # elif (pos_x >= 7000) and (pos_x <= 1000) and (pos_y >= 6000) and (pos_y <= 10000):
            #     set_angle =-1*np.pi * 0.5
            #
            # elif (pos_x >= 7000) and (pos_x <= 10000) and (pos_y >= 0) and (pos_y <= 4000):
            #     set_angle = np.pi * 0.5


            pivot_1=(5000,7500)
            pivot_2=(5000,2500)

            dist_1=distance.euclidean(pivot_1,(pos_x, pos_y))
            dist_2 = distance.euclidean(pivot_2, (pos_x, pos_y))

            if dist_1<=dist_2:

                set_angle = math.atan2(pos_y - 7500.01, pos_x - 5000.01) + np.pi * 0.5

            else:

                set_angle = math.atan2(pos_y - 2500.01, pos_x - 5000.01) + np.pi * 0.5

            # set_angle = math.atan2(pos_y - 5000.1, pos_x - 5000.01) + np.pi * 0.5

            if np.sum(mask_matrix)==0:

                print("what is oging heere " + str((pos_x, pos_y)))


            try:
                mu_r = 10000 / np.sum(mask_matrix)
                sigma_r = np.log(500) / np.sum(mask_matrix)
                mu_angle = set_angle / np.sum(mask_matrix)
                sigma_angle = np.log(0.05) / np.sum(mask_matrix)

            except:

                pass
                # mu_r = 300 / np.sum(mask_matrix)
                # sigma_r = np.log(100) / np.sum(mask_matrix)
                # mu_angle = set_angle / np.sum(mask_matrix)
                # sigma_angle = np.log(0.15) / np.sum(mask_matrix)

            self.theta_u_r[mask_matrix] = mu_r
            self.theta_sigma_r[mask_matrix] = sigma_r
            self.theta_u_angle[mask_matrix] = mu_angle
            self.theta_sigma_angle[mask_matrix] = sigma_angle

    def estimated_val_function(self,feature_vector):

        v_hat=np.sum(np.multiply(feature_vector,self.w))

        return v_hat

    def sample_step_length(self,feature_vector):


        mu_r=np.sum(np.multiply(self.theta_u_r, feature_vector))
        sigma_r=np.exp(np.sum(np.multiply(self.theta_sigma_r, feature_vector)))

        if sigma_r>500:

            sigma_r=100

        lenght=norm.rvs(loc=mu_r, scale=sigma_r)

        if np.abs(lenght)>9000:

            lenght=9000

        if lenght<0:

            lenght=-1*lenght

        return (lenght, mu_r, sigma_r)

    def sample_step_angle(self, feature_vector):

        mu_angle = np.sum(np.multiply(self.theta_u_angle, feature_vector))
        sigma_angle = np.exp(np.sum(np.multiply(self.theta_sigma_angle, feature_vector)))
        angle = norm.rvs(loc=mu_angle, scale=sigma_angle)

        return (angle, mu_angle, sigma_angle)

    def take_action(self, pos_x,pos_y , lenght, angle):

        new_pos_x=pos_x + lenght*np.cos(angle)
        new_pos_y=pos_y + lenght*np.sin(angle)



        if new_pos_x > self.L:
            new_pos_x = 2 * self.L - new_pos_x

        if new_pos_x < 0:
            new_pos_x = -1 * new_pos_x

        if new_pos_y > self.L:
            new_pos_y = 2 * self.L - new_pos_y

        if new_pos_y < 0:
            new_pos_y = -1 * new_pos_y

        self.pos_x=new_pos_x
        self.pos_y=new_pos_y

        self.path_x.append(self.pos_x)
        self.path_y.append(self.pos_y)

        self.distance_traveled+= distance.euclidean((pos_x, pos_y),(new_pos_x, new_pos_y))

        return (new_pos_x, new_pos_y)

    def get_reward(self, pos_x, pos_y,new_pos_x, new_pos_y):


        index_x=int(np.floor(pos_x/50))
        index_y=int(np.floor(pos_y/50))

        print("rewards in cell : " + str(self.target_grid[index_x, index_y]))
        n_targets=self.target_grid[index_x, index_y]
        self.collected_targets+=n_targets
        self.target_grid[index_x, index_y]=0

        delta_distance=np.log(distance.euclidean((pos_x, pos_y),(new_pos_x, new_pos_y)))

        reward= n_targets*10-delta_distance

        return reward

    def calculate_delta(self,reward,v_s, v_s_prime):

        delta=reward+ v_s_prime - v_s
        return  delta

    def update_w_vector(self, delta, feature_vector):

        add_comp=self.alpha_w*delta*feature_vector
        self.w= np.add(self.w,add_comp)

    def update_theta_u_r(self,delta, mu_r,sigma_r, step_length, feature_vector):

        nabla_ln=((step_length-mu_r)/(sigma_r**2))*feature_vector
        add_comp=self.alpha_u_r*delta*nabla_ln
        self.theta_u_r=np.add(self.theta_u_r, add_comp)
        self.updates_theta_u_r.append(self.alpha_u_r*delta)


    def update_theta_sigma_r(self,delta, mu_r,sigma_r, step_length, feature_vector):

        nabla_ln = (((step_length - mu_r) / (sigma_r))**2) * feature_vector
        add_comp = self.alpha_sigma_r * delta * nabla_ln
        self.theta_sigma_r = np.add(self.theta_sigma_r, add_comp)
        self.updates_theta_sigma_r.append(self.alpha_sigma_r * delta)

    def update_theta_u_angle(self,delta, mu_angle,sigma_angle, step_angle, feature_vector):

        nabla_ln=((step_angle-mu_angle)/(sigma_angle**2))*feature_vector
        add_comp=self.alpha_u_angle*delta*nabla_ln
        self.theta_u_angle=np.add(self.theta_u_angle, add_comp)
        self.updates_theta_u_angle.append(self.alpha_u_angle*delta)

    def update_theta_sigma_angle(self,delta, mu_angle,sigma_angle, step_angle, feature_vector):

        nabla_ln = (((step_angle - mu_angle) / (sigma_angle)) ** 2) * feature_vector
        add_comp = self.alpha_sigma_angle * delta * nabla_ln
        self.theta_sigma_angle = np.add(self.theta_sigma_angle, add_comp)
        self.updates_theta_sigma_angle.append( self.alpha_sigma_angle * delta )

    def search_efficiency(self):

        return self.collected_targets/self.distance_traveled

    def reset_agent(self,target_grid):

        self.pos_x = 5000
        self.pos_y = 5000
        self.collected_targets = 0
        self.distance_traveled = 0
        self.path_x = [self.pos_x]
        self.path_y = [self.pos_y]
        self.updates_theta_u_r = []
        self.updates_theta_sigma_r = []
        self.updates_theta_u_angle = []
        self.updates_theta_sigma_angle = []
        self.target_grid = target_grid

    def plot_path(self, target_location):

        coord_x=[i[0] for i in target_location]
        coord_y=[j[1] for j in target_location]
        plt.plot(coord_x, coord_y, 'o')
        plt.plot(self.path_x, self.path_y)
        plt.plot
        plt.show()

    def terminal_state(self,pos_x, pos_y):

        if (((pos_y<=5100) and (pos_y>=4900)) and ((pos_x>=5100) and (pos_x<=5300))):

            mask_matrix = self.pos_to_feature(pos_x, pos_y).astype(bool)
            self.w[mask_matrix]=0

            return True
        else:
            return False




if __name__ == "__main__":


    E=1000
    targets = np.loadtxt('target_large.csv', delimiter=',')
    target_location = [(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])]

    enviroment = Enviroment( target_location)
    target_grid=enviroment.targets
    #
    # plt.imshow(target_grid, cmap='hot', interpolation='nearest')
    # plt.show()
    agent=Agent(copy.deepcopy(enviroment.targets))
    print("todos targets " + str(np.sum(agent.target_grid)))
    agent.initialize_thetas()


    eta=[]
    for episode in range(E):

        time_t=0

        terminal_state = False

        while not terminal_state:



            pos_x = agent.pos_x
            pos_y = agent.pos_y

            feature_vector= agent.pos_to_feature(pos_x,pos_y)
            step_length, mu_r, sigma_r= agent.sample_step_length(feature_vector)
            step_angle, mu_angle, sigma_angle = agent.sample_step_angle(feature_vector)
            new_pos_x, new_pos_y=agent.take_action(pos_x, pos_y, step_length, step_angle)
            print(" at position " + str((pos_x, pos_y)))
            print("mean sigma l " + str((mu_r, sigma_r)))
            print("mean sigma ange " + str((mu_angle, sigma_angle)))
            print(" sample l and theta " + str((step_length, step_angle)))
            # time.sleep(1)
            reward=agent.get_reward(pos_x, pos_y,new_pos_x, new_pos_y)
            v_s=agent.estimated_val_function(feature_vector)
            feature_vector_prime=agent.pos_to_feature(new_pos_x,new_pos_y)
            v_s_prime=agent.estimated_val_function(feature_vector_prime)
            delta=agent.calculate_delta(reward, v_s, v_s_prime)
            # print(" Delta: "  +str(delta))
            # print("Rewad " + str(reward))
            agent.update_w_vector(delta, feature_vector)
            agent.update_theta_u_r(delta, mu_r, sigma_r, step_length, feature_vector)
            agent.update_theta_sigma_r(delta, mu_r, sigma_r,step_length, feature_vector)
            agent.update_theta_u_angle(delta, mu_angle, sigma_angle, step_angle, feature_vector)
            agent.update_theta_sigma_angle(delta,mu_angle,sigma_angle, step_angle, feature_vector)
            print("Time step " +str(time_t) + " targets left " + str(np.sum(agent.target_grid)))
            # time.sleep(1)
            terminal_state = agent.terminal_state(new_pos_x, new_pos_y)



            # time.sleep(0.5)

        eta.append(agent.search_efficiency())
        print("search effciency: "  +str(agent.search_efficiency()))
        print("episode " + str(episode))
        # time.sleep(1)

        if episode!=E-1:
            agent.reset_agent(copy.deepcopy(enviroment.targets))


        print("todos targets " + str(np.sum(agent.target_grid)))

    agent.plot_path(target_location)



# def plot_targets(target_location):
#
#
#     xpoints=[i[0] for i in target_location]
#     ypoints=[i[1] for i in target_location]
#     plt.plot(xpoints, ypoints, 'o')
#     plt.show()
#
# plot_targets(target_location)
#
# #
# # targetnp.loadtxt("targets_x.out")
#
#
#
# def plot_search_efficiency(eta):
#
#     time=list(range(1950))
#     n_eta = []
#
#     for i in range(1950):
#         n_eta.append(np.mean(eta[i:i + 50]))
#
#     plt.plot(time,n_eta )
#     plt.show()
#
#
#
# plot_search_efficiency(eta)
#
#
#
#
# def funct_stoc(x):
#
#     y=1e-4*np.log(x)-0.0006
#     y+=norm.rvs(loc=y, scale=3e-5)
#
#     if y<0:
#
#         y=np.abs(norm.rvs(loc=y, scale=2.5e-5))
#
#     return y
#
#
# x_0=list(np.linspace(100, 5000, num=500))
# y_stoc=[funct_stoc(x) for x in x_0]
# y_det=[funct_det(x) for x in x_0]
#
# plt.plot(x_0, y_det)
# plt.plot(x_0, y_stoc)
# plt.ticklabel_format(axis='y',  style="sci", scilimits=(0,1))
# plt.legend(["deterministic", "stochastic"], loc ="lower right")
# plt.xlabel('Episode ')
# plt.ylabel('Search efficiency')
# plt.show()
#
#
#
# plt.plot(x_0, y_0)
# plt.show()
#
#
# #Section to create sample of targets that allows to calibrate the weights
#

patches_x=norm.rvs(loc=5000, scale=2000, size=10)
patches_y=norm.rvs(loc=5000, scale=2000, size=10)

targets_x=[]
targets_y=[]

targets_x1=[]
targets_y1=[]


for i in range(10):

    targets_x=targets_x+list(norm.rvs(loc=patches_x[i], scale=450, size=50))
    targets_y = targets_y + list(norm.rvs(loc=patches_y[i], scale=450, size=50))
    targets_x1 = targets_x1 + list(norm.rvs(loc=patches_x[i], scale=200, size=50))
    targets_y1= targets_y1 + list(norm.rvs(loc=patches_y[i], scale=200, size=50))

# plt.plot(targets_x,targets_y, 'o')
plt.plot(targets_x1,targets_y1, 'o', color='red')
plt.plot
plt.show()

#
# np.savetxt('targets_x.out',targets_x , delimiter=',')
# np.savetxt('targets_y.out',targets_y , delimiter=',')
#
# targets_x=np.loadtxt("targets_x.out")
# targets_y=np.loadtxt("targets_y.out")
#
# x_samples=uniform.rvs(0,10000, size=8000)
# y_samples=uniform.rvs(0,10000, size=8000)
#
# X_matrix = np.zeros((8000, 4000))
# Y_matrix = np.zeros((8000, 1))
#
# for row in range(8000):
#
#     pos_x=x_samples[row]
#     pos_y=y_samples[row]
#     distances_to_patches=[distance.euclidean((pos_x, pos_y),(patches_x[i], patches_y[i]))for i in range(10)]
#
#     mask_matrix=agent.pos_to_feature(pos_x, pos_y).astype(bool)
#     set_angle=math.atan2(pos_y-5000,pos_x-5000) + np.pi*0.5
#     mu_r=300/ np.sum(mask_matrix)
#     sigma_r=100/ np.sum(mask_matrix)
#     mu_angle=set_angle/ np.sum(mask_matrix)
#     sigma_angle=np.log(0.1)/ np.sum(mask_matrix)
#     agent.theta_u_r[mask_matrix]=mu_r
#     agent.theta_sigma_r[mask_matrix] = sigma_r
#     agent.theta_u_angle[mask_matrix] = mu_angle
#     agent.theta_sigma_angle[mask_matrix] = sigma_angle
#
#
#
#
# x_s=agent.pos_to_feature(7500,2500)
#
# np.sum(np.multiply(x_s, agent.theta_u_r))
# np.sum(np.multiply(x_s, agent.theta_sigma_r))
# np.sum(np.multiply(x_s, agent.theta_u_angle))
# np.sum(np.multiply(x_s, agent.theta_sigma_angle))
#
#
# np.linalg.inv(np.matmul(X_matrix.T, X_matrix))
#
#
# m = np.matrix([[1,2], [3,4]])
# m.flatten('F')
#
# math.degrees(1.23)
#
# a=[1,5,9]
#
#
#
#
# np.arctan(-2/3) + np.pi*0.5
#
#
# math.atan2(2000,-1000) + np.pi*0.5
#
# (240.20522321409362, 4873.056844473154)
#
#
#
# fv=agent.pos_to_feature(477, 6215)
# agent.sample_step_length(fv)
#




