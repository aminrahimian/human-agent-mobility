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
        agent_position=(pos_x,pos_y)

        cont=0
        initial_range=len(self.target_location)
        removal=set({})
        list_target_location=list(self.target_location)

        for i in range(initial_range):

            if distance.euclidean(agent_position, list_target_location[i])<= radius_detection:

                removal.add(list_target_location[i])
                cont+=1

            else:

                pass

        self.target_location=self.target_location.difference(removal)

        if cont > 0:

            R = 100 * cont
            print("Total targets : " + str(cont))

        else:

            R=-10


        return (R, cont)

class Agent:

    def __init__(self, radius_coarse, L, T,radius_detection):

        self.L = L
        self.T = T
        self.pos_x = L/2
        self.pos_y = L/2
        self.pos_x_prime=L/2
        self.pos_y_prime=L/2

        # centers of the circles for coarse coding
        # center each 1000 m ------------------------------------------

        coarse_centers = 1000 * np.arange(0,11,1)
        self.c_1 = list(itertools.product(coarse_centers, coarse_centers))
        self.c_2 = list(itertools.product(coarse_centers, coarse_centers))
        self.c_0 = [(L / 2, L / 2)]
        self.c = self.c_0 + self.c_1 + self.c_2
        self.w = np.array(len(self.c)*[0])
        self.theta_mu = np.array(len(self.c)*[0] + len(self.c)*[0])
        self.theta_mean_beta = np.array(len(self.c) * [0])
        self.theta_sd_beta = np.array(len(self.c) * [0.3])  # an approximation of 1 standard deviation

        self.alpha_mean = 5e-2
        self.alpha_sd = 5e-4

        #---------------------------------------------------------------

        self.radius_coarse = radius_coarse
        self.radius_detection=radius_detection
        self.cum_targets = 0
        self.alpha_w = 1e-2
        self.alpha_theta= 1e-4
        self.gamma = 1
        self.time_step = 0
        self.parameter=2
        self.A=[]
        self.t=0
        self.path_x=[self.pos_x]
        self.path_y=[self.pos_y]
        self.targets_collected=[]

    def feature_vector(self, pos_x, pos_y):

        """
        :param pos_x:
        :param pos_y:
        :return: array of vector of 0-1
        """

        if self.t==0:

            inside_circle = np.array([1] + len(self.c_1) * [0] + len(self.c_2) * [0])
            # print(" init state")

        elif (self.t>0) & (self.t+1<=self.T):

            # print(">1")
            actual_pos = len(self.c_1) * [(pos_x, pos_y)]
            iter = len(self.c_1)
            distances = np.array([distance.euclidean(self.c_1[i], actual_pos[i]) for i in range(iter)])
            inside_circle_1 = (distances <= self.radius_coarse)
            inside_circle_1 = list(inside_circle_1.astype(int))

            inside_circle=[0]+ inside_circle_1+ len(self.c_1) * [0]


        else:

            # print("Final grid ")
            actual_pos = len(self.c_1) * [(pos_x, pos_y)]
            iter = len(self.c_1)
            distances = np.array([distance.euclidean(self.c_1[i], actual_pos[i]) for i in range(iter)])
            inside_circle_1 = (distances <= self.radius_coarse)
            inside_circle_1 = list(inside_circle_1.astype(int))

            inside_circle = [0] + len(self.c_1) * [0]+inside_circle_1
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

    def sample_parameter(self,feature_vector):

        """
        :param pos_x: position x
        :param pos_y: position y
        :return: {2,3}
        """

        # calculate h(s,a theta)

        partial_x_sa=list(feature_vector)
        zeros_x_sa=len(partial_x_sa)*[0]
        x_sa_mu_2 = partial_x_sa + zeros_x_sa
        x_sa_mu_3 =  zeros_x_sa + partial_x_sa
        h_mu_2 = np.dot(self.theta_mu, x_sa_mu_2)
        h_mu_3 = np.dot(self.theta_mu, x_sa_mu_3)
        p_mu_2 = np.exp(h_mu_2)/(np.exp(h_mu_2) +np.exp(h_mu_3))
        # print("Probability of u_2 : " +str(p_mu_2))
        self.parameter=random.choices([2,3], weights=[p_mu_2, 1-p_mu_2], k=1)[0]

        return self.parameter

    def probability_actions(self, pos_x, pos_y):

        partial_x_sa = list(self.feature_vector(pos_x, pos_y))
        zeros_x_sa = len(partial_x_sa) * [0]
        x_sa_mu_2 = partial_x_sa + zeros_x_sa
        x_sa_mu_3 = zeros_x_sa + partial_x_sa
        h_mu_2 = np.dot(self.theta, x_sa_mu_2)
        h_mu_3 = np.dot(self.theta, x_sa_mu_3)

        print("Weights for u2 and u3 " + str((h_mu_2, h_mu_3)))

        p_mu_2 = np.exp(h_mu_2) / (np.exp(h_mu_2) + np.exp(h_mu_3))

        return p_mu_2

    def sample_step_length(self,mu):

        alpha= mu -1
        V = np.random.uniform(-np.pi*0.5, 0.5*np.pi, 1)[0]
        W = expon.rvs(size=1)[0]
        cop1=(np.sin(alpha*V))/((np.cos(V))**(1/alpha))
        cop2=((np.cos((1-alpha)*V))/W)**(((1-alpha)/alpha))
        elle=cop1*cop2

        if elle<=0:

            elle = 10*min(500,-1*elle)
            step_l = max(25, elle)

        else:

            elle = 10*min(500, elle)
            step_l = max(25, elle)

        # print("Taking this value:  " + str(elle))

        # angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # print("angle vel :" + str(angle))
        # vel_x = 10*elle * np.cos(angle)
        # vel_y = 10*elle * np.sin(angle)

        return step_l

    def nabla_pi_sa(self, feature_vector, mu, delta):

        partial_x_sa = list(feature_vector)
        zeros_x_sa = len(partial_x_sa) * [0]
        x_sa_mu_2 = partial_x_sa + zeros_x_sa
        x_sa_mu_3 = zeros_x_sa + partial_x_sa

        h_mu_2 = np.dot(self.theta_mu, x_sa_mu_2)
        h_mu_3 = np.dot(self.theta_mu, x_sa_mu_3)

        p_mu_2 = np.exp(h_mu_2) / (np.exp(h_mu_2) + np.exp(h_mu_3))
        p_mu_3 = 1 - p_mu_2

        # print("p_mu_2, p_mu_3: " + str((p_mu_2, p_mu_3)))

        cum_soft_max = p_mu_2 * np.array(x_sa_mu_2) + \
                       p_mu_3 * np.array(x_sa_mu_3)

        if mu == 2:

            nabla = x_sa_mu_2 - cum_soft_max

        elif mu==3:

            nabla = x_sa_mu_3 - cum_soft_max

        return delta*nabla

    def next_state(self,l,beta):

        vel_x = l * np.cos(beta)
        vel_y = l * np.sin(beta)
        pos_x_prime = self.pos_x + vel_x
        pos_y_prime = self.pos_y + vel_y

        if pos_x_prime > self.L:
            pos_x_prime = 2 * self.L - pos_x_prime

        if pos_x_prime < 0:
            pos_x_prime = -1 * pos_x_prime

        if pos_y_prime > self.L:
            pos_y_prime = 2 * self.L - pos_y_prime

        if pos_y_prime < 0:
            pos_y_prime = -1 * pos_y_prime

        return (pos_x_prime, pos_y_prime)

    def update_state(self, pos_x_prime, pos_y_prime):

        self.pos_x = pos_x_prime
        self.pos_y = pos_y_prime
        self.path_x.append(self.pos_x)
        self.path_y.append(self.pos_y)
        self.t+=1

    def update_w(self, delta, pos_x, pos_y):

        x_s=self.feature_vector(pos_x, pos_y)
        update=np.where(x_s>0, delta*self.alpha_w*x_s, x_s)
        self.w=np.add(self.w, update)

        pass


    def sample_angle(self, feature_vector):

        beta_mean=np.dot(self.theta_mean_beta,feature_vector)
        beta_sd=np.exp(np.dot(self.theta_sd_beta,feature_vector))
        beta=norm.rvs(beta_mean, beta_sd, size=1)[0]
        print("Normal distribution with mean " + str(beta_mean) + " and sd : " +str(beta_sd))

        return (beta, beta_mean, beta_sd)

    def update_theta_mean_beta(self,delta,beta_i, beta_mean, beta_sd, feature_vector):

        gradient_mean=self.alpha_mean*delta*(beta_i-beta_mean)*(1/beta_sd**2)*feature_vector
        print("Change in mean: " + str(self.alpha_mean*delta*(beta_i-beta_mean)*(1/beta_sd**2)))
        self.theta_mean_beta=np.add(self.theta_mean_beta, gradient_mean)

        pass

    def update_theta_sd_beta(self, delta,beta_i, beta_mean, beta_sd, feature_vector):

        gradient_sd = self.alpha_sd *delta* (((beta_i-beta_mean)**2/(beta_sd)**2)-1) * feature_vector
        print("Change in sd: " +str(self.alpha_sd *delta* (((beta_i-beta_mean)**2/(beta_sd)**2)-1) ) )
        self.theta_sd_beta = np.add(self.theta_sd_beta, gradient_sd)

        pass


    def update_theta_mu(self, mu, delta, feature_vector):

        self.theta_mu=np.add(self.theta_mu, self.alpha_theta*self.nabla_pi_sa(feature_vector,mu, delta))

        pass

    def delta(self, feature_vector,feature_vector_prime, reward):

        v_hat=np.dot(self.w, feature_vector)
        v_hat_prime= np.dot(self.w, feature_vector_prime)

        return reward+v_hat_prime-v_hat

    def delta_T(self, feature_vector, reward):

        v_hat = np.dot(self.w, feature_vector)

        return reward - v_hat

    def generate_one_step(self,mu,beta,enviroment):

        self.update_next_state(mu)

        # print("current position: " +str((a1.pos_x, a1.pos_y)))
        # print("next  position: " + str((a1.pos_x_prime, a1.pos_y_prime)))

        self.update_state()

        number_targets = enviroment.collected_targets(self)
        self.targets_collected.append(number_targets)


        return 1

    def reset_agent(self):

        self.pos_x = L / 2
        self.pos_y = L / 2
        self.pos_x_prime = L / 2
        self.pos_y_prime = L / 2
        self.t=0
        self.path_x = [self.pos_x]
        self.path_y = [self.pos_y]
        self.targets_collected=[]
        self.A = []

    def plot_path(self,targets):

        """
        Plot a search path given the location of the targets as 2D np array
        :param targets:
        :return:
        """

        plt.plot(self.path_x, self.path_y, '--o', color='#65C7F5')
        plt.plot(targets[:, 0], targets[:, 1], 'ko')
        plt.show()

    def save_weights(self):

        np.savetxt('w.csv', self.w, delimiter=',')
        np.savetxt('theta_mu.csv', self.theta_mu, delimiter=',')
        np.savetxt('theta_mean_beta.csv', self.theta_mean_beta, delimiter=',')
        np.savetxt('theta_sd_beta.csv', self.theta_sd_beta, delimiter=',')

        pass

    def load_weights(self):

        w=np.loadtxt('w.csv', self.w, delimiter=',')
        theta_mu=np.loadtxt('theta_mu.csv', self.theta_mu, delimiter=',')
        theta_mean_beta= np.loadtxt('theta_mean_beta.csv', self.theta_mean_beta, delimiter=',')
        theta_sd_beta= np.loadtxt('theta_sd_beta.csv', self.theta_sd_beta, delimiter=',')

        return (w,theta_mu,theta_mean_beta,theta_sd_beta)


#Initialization values

if __name__ == "__main__":

    n_targets=[]
    radius_coarse=800
    L=10000
    T=1000
    radius_detection=25

    # Create robot and enviroment objects

    robot=Agent(radius_coarse,L,T,radius_detection)
    targets = np.loadtxt('target_large.csv', delimiter=',')
    target_location = set([(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])])
    env = Enviroment(L, target_location)
    np.mean([robot.sample_step_length(3) for i in range(1000)])
    # test iteration

    while robot.t!=robot.T:

        pos_x=robot.pos_x
        pos_y=robot.pos_y
        feature_vector=robot.feature_vector(pos_x, pos_y)
        mu=robot.sample_parameter(feature_vector)
        l=robot.sample_step_length(mu)
        beta_i, beta_mean, beta_sd=robot.sample_angle(feature_vector)
        pos_x_prime, pos_y_prime = robot.next_state(l,beta_i)
        feature_vector_prime=robot.feature_vector(pos_x_prime, pos_y_prime)
        reward,_=env.collected_targets(pos_x_prime, pos_y_prime, radius_detection)
        delta=robot.delta(feature_vector, feature_vector_prime, reward)
        robot.update_w(delta,pos_x, pos_y)
        robot.update_theta_mu(mu, delta, feature_vector)
        robot.update_theta_mean_beta(delta,beta_i, beta_mean, beta_sd,feature_vector)
        robot.update_theta_sd_beta(delta,beta_i, beta_mean, beta_sd, feature_vector)

        # print("From: " +str((pos_x, pos_y))+" to: " + str((pos_x_prime, pos_y_prime)) +
        #       "\n l : " +str(l) + " beta:  " +str(beta_i) + " mu:" + str(mu) + " Reward: " +str(reward))
        robot.update_state(pos_x_prime, pos_y_prime)
        robot.t
        # time.sleep(1)

    robot.save_weights()
    w, theta_mu, theta_mean_beta, theta_sd_beta=robot.load_weights()

    #Learning block


# plot path
targets = np.loadtxt('target_large.csv', delimiter=',')
robot.plot_path(targets)

# x_t=targets[:,0]
# y_t=targets[:,1]
#
#
# def plot_heat_map():
#
#     rng = np.random.default_rng()
#     x = 10000 *np.random.uniform(0,1,2000)
#     y = 10000 *np.random.uniform(0,1,2000)
#     z=[a1.probability_actions(vector_x[i], vector_y[i]) for i in range(2000)]
#     X = np.linspace(min(x), max(x))
#     Y = np.linspace(min(y), max(y))
#     X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
#     interp = NearestNDInterpolator(list(zip(x, y)), z)
#     Z = interp(X, Y)
#     plt.pcolormesh(X, Y, Z, shading='auto')
#     plt.plot(x_t, y_t, "ok", label="input point", color="red")
#     # plt.legend()
#     # plt.colorbar()
#     # plt.axis("equal")
#     plt.show()
#
