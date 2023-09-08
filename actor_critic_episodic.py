import numpy as np
from scipy.stats import norm
from scipy.stats import bernoulli
import itertools
from scipy.spatial import distance
import copy
import pickle
import matplotlib.pyplot as plt
import time
from scipy.stats import pareto


# generate targets.
class Enviroment:
    def __init__(self, L, target_location):

        self.L = L
        self.target_location=target_location


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

    def collected_targets(self, agent):

        """ Return the number of targets given the historical
        position of the agent
        """
        path_x = agent.path_x
        path_y = agent.path_y

        agent_path=[(path_x[i], path_y[i]) for i in range(len(path_x))]

        target_combinations = list(itertools.product(agent_path, self.target_location))

        target_dist = [distance.euclidean(target_combinations[i][0], target_combinations[i][1]) for i in
                       range(len(target_combinations))]

        return sum(np.array(target_dist) <= agent.radius_detection)

class Agent:

    def __init__(self, radius_coarse, L, T,radius_detection, freq_sampling,episodes):

        self.L = L
        self.T = T
        self.pos_x = L/2
        self.pos_y = L/2
        self.pos_x_prime=L/2
        self.pos_y_prime=L/2
        grid_line = 1000 * np.arange(0.5, 10.5, 1)
        self.center_lay1 = list(itertools.product(grid_line, grid_line))
        self.center_lay2 = list(itertools.product(grid_line, grid_line))
        self.center_init = [(L/2, L/2)]
        self.centers = self.center_init + self.center_lay1 + self.center_lay2
        self.weigths_vector = np.array([100] + len(self.center_lay1) * [100] + len(self.center_lay2) * [0])
        self.theta_mu_x_vector=  [0.1] + len(self.center_lay1) * [0.1] + len(self.center_lay2) * [0]
        self.theta_sd_x_vector = [0.01] + len(self.center_lay1) * [0.01] + len(self.center_lay2) * [0]
        self.theta_mu_y_vector = [0.01] + len(self.center_lay1) * [0.01] + len(self.center_lay2) * [0]
        self.theta_sd_y_vector = [0.01] + len(self.center_lay1) * [0.01] + len(self.center_lay2) * [0]
        self.theta=self.theta_mu_x_vector+self.theta_mu_x_vector
        self.radius_coarse = radius_coarse
        self.radius_detection=radius_detection
        self.freq_sampling=freq_sampling
        self.cum_targets = 0
        self.alpha = 0.1
        self.gamma = 1
        self.time_step = 0
        self.episodes=episodes
        self.parameter=2
        self.episode=0
        self.path_x=[self.pos_x]
        self.path_y=[self.pos_y]
        self.targets_collected=[0,0]

    def feature_vector(self, pos_x, pos_y):

        if self.episode==0:

            inside_circle = np.array([1] + len(self.center_lay1) * [0] + len(self.center_lay2) * [0])
            # print(" init state")

        elif (self.episode>0) & (self.episode+1<=self.episodes):

            # print(">1")
            actual_pos = len(self.center_lay1) * [(pos_x, pos_y)]
            iter = len(self.center_lay1)
            distances = np.array([distance.euclidean(self.center_lay1[i], actual_pos[i]) for i in range(iter)])
            inside_circle_1 = (distances <= self.radius_coarse)
            inside_circle_1 = list(inside_circle_1.astype(int))

            inside_circle=[0]+ inside_circle_1+ len(self.center_lay1) * [0]


        else:

            print("Final grid ")
            actual_pos = len(self.center_lay1) * [(pos_x, pos_y)]
            iter = len(self.center_lay1)
            distances = np.array([distance.euclidean(self.center_lay1[i], actual_pos[i]) for i in range(iter)])
            inside_circle_1 = (distances <= self.radius_coarse)
            inside_circle_1 = list(inside_circle_1.astype(int))

            inside_circle = [0] + len(self.center_lay1) * [0]+inside_circle_1
            # print(" feature vector " +str(inside_circle) )


        return np.array(inside_circle)
    def sample_action_continous(self):

        action_mean_x = np.dot(self.theta_mu_x_vector,
                               self.feature_vector(self.pos_x, self.pos_y))

        action_sd_x = np.exp(np.dot(self.theta_mu_x_vector,
                                    self.feature_vector(self.pos_x, self.pos_y)))

        action_mean_y = np.dot(self.theta_mu_y_vector,
                               self.feature_vector(self.pos_x, self.pos_y))

        action_sd_y = np.exp(np.dot(self.theta_mu_y_vector,
                                    self.feature_vector(self.pos_x, self.pos_y)))

        vel_x = norm.rvs(loc=action_mean_x, scale=action_sd_x, size=1)[0]
        vel_y = norm.rvs(loc=action_mean_y, scale=action_sd_y, size=1)[0]

        return (vel_x, vel_y)
    def sample_parameter(self):

        #calculate h(s,a theta)

        partial_x_sa=list(self.feature_vector(self.pos_x, self.pos_y))
        zeros_x_sa=len(partial_x_sa)*[0]
        x_sa_mu_2 = partial_x_sa + zeros_x_sa
        x_sa_mu_3 =  zeros_x_sa + partial_x_sa
        h_mu_2 = np.dot(self.theta, x_sa_mu_2)
        h_mu_3 = np.dot(self.theta, x_sa_mu_3)
        p_mu_2 = np.exp(h_mu_2)/(np.exp(h_mu_2) +np.exp(h_mu_3))

        # print("Probability of u_2 : " +str(p_mu_2))

        is_mu_2=bernoulli.rvs(p_mu_2, size=1)[0]

        if is_mu_2==1:

            self.parameter=2

        else:

            self.parameter=3

    def sample_action(self):

        self.sample_parameter()
        mu=self.parameter-1

        l=pareto.rvs(mu, scale=250,size=1)[0]
        angle=np.random.uniform(0,2*np.pi,1)[0]
        # print("angle vel :" + str(angle))
        vel_x = l*np.cos(angle)
        vel_y = l*np.sin(angle)

        return (vel_x, vel_y)

    def nabla_pi_sa(self):

        partial_x_sa = list(self.feature_vector(self.pos_x, self.pos_y))
        zeros_x_sa = len(partial_x_sa) * [0]
        x_sa_mu_2 = partial_x_sa + zeros_x_sa
        x_sa_mu_3 = zeros_x_sa + partial_x_sa

        h_mu_2 = np.dot(self.theta, x_sa_mu_2)
        h_mu_3 = np.dot(self.theta, x_sa_mu_3)

        p_mu_2 = np.exp(h_mu_2) / (np.exp(h_mu_2) + np.exp(h_mu_3))
        p_mu_3 = 1- p_mu_2

        # print("p_mu_2, p_mu_3: " + str((p_mu_2, p_mu_3)))

        cum_soft_max=p_mu_2*np.array(x_sa_mu_2)+\
                     p_mu_3*np.array(x_sa_mu_3)

        if self.parameter==2:

            nabla = x_sa_mu_2 - cum_soft_max

        else:

            nabla = x_sa_mu_3 - cum_soft_max

        return  nabla

    def update_next_state(self,action):

        vel_x=action[0]
        vel_y=action[1]
        self.pos_x_prime = self.pos_x + vel_x * 1
        self.pos_y_prime = self.pos_y + vel_y * 1

        if self.pos_x_prime > self.L:
            self.pos_x_prime = 2 * self.L - self.pos_x_prime

        if self.pos_x_prime < 0:
            self.pos_x_prime = -1 * self.pos_x_prime

        if self.pos_y_prime > self.L:
            self.pos_y_prime = 2 * self.L - self.pos_y_prime

        if self.pos_y_prime < 0:
            self.pos_y_prime = -1 * self.pos_y_prime

    def update_state(self):

        self.pos_x = self.pos_x_prime
        self.pos_y = self.pos_y_prime
        self.path_x.append(self.pos_x)
        self.path_y.append(self.pos_y)
        self.episode+=1

    def updates_weigts(self, action, enviroment):


        self.update_next_state(action)
        reward=self.targets_collected[-1]-self.targets_collected[-2]
        print("current position: " +str((a1.pos_x, a1.pos_y)))
        print("next  position: " + str((a1.pos_x_prime, a1.pos_y_prime)))
        x_s = self.feature_vector(self.pos_x, self.pos_y)
        x_s_prime = self.feature_vector(self.pos_x_prime, self.pos_y_prime)
        current_s = np.dot(x_s, self.weigths_vector)
        next_s = self.gamma * np.dot(x_s_prime, self.weigths_vector)
        # print("v_(S, w) : "  +str(current_s))
        # print("v_(S',w) : " + str(next_s))

        if (self.episode+2 <=self.episodes):

            delta = reward + next_s - current_s

            self.weigths_vector = np.array(self.weigths_vector) + \
                                  np.array(self.alpha * delta * x_s)

            self.theta = self.theta + \
                         self.alpha * delta * self.nabla_pi_sa()

        elif (self.episode+1 ==self.episodes):

            delta = reward  - current_s

            self.weigths_vector = np.array(self.weigths_vector) + \
                                  np.array(self.alpha * delta * x_s)

            self.theta = self.theta + \
                         self.alpha * delta * self.nabla_pi_sa()

        else:

            pass

        self.update_state()

        self.targets_collected.append(env.collected_targets(self))

        return delta

    def reset_agent(self):

        self.pos_x = L / 2
        self.pos_y = L / 2
        self.pos_x_prime = L / 2
        self.pos_y_prime = L / 2
        self.episode=0
        self.path_x = [self.pos_x]
        self.path_y = [self.pos_y]
        self.targets_collected=[0,0]

    def plot_path(self,targets):

        """
        Plot a search path given the location of the targets as 2D np array
        :param targets:
        :return:
        """

        plt.plot(a1.path_x, a1.path_y, '--o', color='#65C7F5')
        plt.plot(targets[:, 0], targets[:, 1], 'ko')
        plt.show()


#Initialization values

radius_coarse=750
L=10000
T=50
radius_detection=25
freq_sampling=1
episodes=25


targets=np.loadtxt('target_large.csv', delimiter=',')
target_location=[(targets[i,0],targets[i,1]) for i in range(targets.shape[0])]

# Define Agent object

a1=Agent(radius_coarse,L,T,radius_detection, freq_sampling, episodes)

env=Enviroment(L, target_location)
#Learning block

for j in range(10):

    # Loop for episodes

    for i in range(25):

        # Loop for one complete episode

        print("==============================")
        print("Episode: " + str(a1.episode))
        action = a1.sample_action()
        print("action " +str (action))
        a1.updates_weigts( action, env)

        # time.sleep(0)
    if j==9:

        break

    a1.reset_agent()


a1.plot_path(targets)
a1.targets_collected
a1.path_y















