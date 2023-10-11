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

        if cont>0:
            R=10*cont

        else:

            R=-1

        return (R, cont)

class Agent:

    def __init__(self, radius_coarse, L, T,radius_detection,w, theta_mu, theta_mean_beta, theta_sd_beta):

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
        self.w = w
        self.theta_mu = theta_mu
        self.theta_mean_beta = theta_mean_beta
        self.theta_sd_beta =theta_sd_beta  # an approximation of 1 standard deviation

        # self.w = np.array(len(self.c) * [0])
        # self.theta_mu = np.array(len(self.c) * [0] + len(self.c) * [0])
        # self.theta_mean_beta = np.array(len(self.c) * [0])
        # self.theta_sd_beta = np.array(len(self.c) * [0.3])  # an approximation of 1 standard deviation

        self.t = 0
        self.alpha_mean = 1e-2/(self.t+1)
        self.alpha_sd = 1e-3

        #---------------------------------------------------------------

        self.step_fixed=30
        self.radius_coarse = radius_coarse
        self.radius_detection=radius_detection
        self.cum_targets = 0
        self.alpha_w = 1e-3
        self.alpha_theta= 1e-5
        self.gamma = 1
        self.time_step = 0
        self.parameter=2
        self.A=[]
        self.t=0
        self.path_x=[self.pos_x]
        self.path_y=[self.pos_y]
        self.targets_collected=0


    def feature_vector(self, pos_x, pos_y):

        """
        :param pos_x:
        :param pos_y:
        :return: array of vector of 0-1
        """

        if self.t==0:

            inside_circle = np.array([1] + len(self.c_1) * [0] + len(self.c_2) * [0])
            # print(" init state")

        elif (self.t>0) & (self.t<self.T):

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
        # print("Normal distribution with mean " + str(beta_mean) + " and sd : " +str(beta_sd))

        return (beta, beta_mean, beta_sd)

    def update_theta_mean_beta(self,delta,beta_i, beta_mean, beta_sd, feature_vector):

        gradient_mean=self.alpha_mean*delta*(beta_i-beta_mean)*(1/beta_sd**2)*feature_vector

        # print("Change in mean: " + str(self.alpha_mean*delta*(beta_i-beta_mean)*(1/beta_sd**2)))
        # print("Len Theta mean bef " +str(len(self.theta_mean_beta)))
        self.theta_mean_beta=np.add(self.theta_mean_beta, gradient_mean)
        # print("Len Theta mean  aft " + str(len(self.theta_mean_beta)))

        pass

    def update_theta_sd_beta(self, delta,beta_i, beta_mean, beta_sd, feature_vector):

        gradient_sd = self.alpha_sd *delta* (((beta_i-beta_mean)**2/(beta_sd)**2)-1) * feature_vector
        # print("Change in sd: " +str(self.alpha_sd *delta* (((beta_i-beta_mean)**2/(beta_sd)**2)-1) ) )

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


    def reset_agent(self):

        # reset should include the already weights
        self.pos_x = L / 2
        self.pos_y = L / 2
        self.pos_x_prime = L / 2
        self.pos_y_prime = L / 2
        self.t=0
        self.path_x = [self.pos_x]
        self.path_y = [self.pos_y]
        self.targets_collected=0
        self.A = []


    def plot_path(self,targets):

        """
        Plot a search path given the location of the targets as 2D np array
        :param targets:
        :return:
        """
        self.t=1

        check_points = np.array(
            [[2000, 3000], [3000, 7000], [4000, 6000], [7000, 4000],
             [9000, 6000], [6000, 8000], [9000, 9000],
             [8000, 2000],
             [1000, 4000],
             [1000,8000],
             [8000,1000]])

        angles=np.zeros((11,1))

        for i in range(11):
            x_s = self.feature_vector(check_points[i, 0], check_points[i, 1])
            # print("FEat vect " + str(x_s))
            media = np.dot(x_s, self.theta_mean_beta)
            sd = np.exp(np.dot(x_s, self.theta_sd_beta))
            angles[i]=media
            plt.arrow(x=check_points[i, 0], y=check_points[i, 1], dx=1000*np.cos(media), dy=1000*np.sin(media), width=.5)
            print("Mean " + str(media) + " Sd " + str(sd) + "for location" + str(
                (check_points[i, 0], check_points[i, 1])))


        plt.plot(self.path_x, self.path_y, '--o', color='#65C7F5')
        plt.plot(targets[:, 0], targets[:, 1], 'ko')

        plt.show()



    def save_weights(self):

        np.savetxt('w.csv', self.w, delimiter=',')
        np.savetxt('theta_mu.csv', self.theta_mu, delimiter=',')
        np.savetxt('theta_mean_beta.csv', self.theta_mean_beta, delimiter=',')
        np.savetxt('theta_sd_beta.csv', self.theta_sd_beta, delimiter=',')

        pass


    def one_episode(self,L,target_location):

        enviroment = Enviroment(L, target_location)

        while self.t != self.T:

            pos_x = self.pos_x
            pos_y = self.pos_y
            feature_vector = self.feature_vector(pos_x, pos_y)
            mu = self.sample_parameter(feature_vector)
            # l = self.sample_step_length(mu)
            l=self.step_fixed
            beta_i, beta_mean, beta_sd = self.sample_angle(feature_vector)
            pos_x_prime, pos_y_prime = self.next_state(l, beta_i)
            feature_vector_prime = self.feature_vector(pos_x_prime, pos_y_prime)
            reward, cont = enviroment.collected_targets(pos_x_prime, pos_y_prime, radius_detection)
            # print("tupla " + str((reward, cont)))
            self.targets_collected += cont
            # print("test " + str(self.targets_collected ))
            delta = self.delta(feature_vector, feature_vector_prime, reward)
            self.update_w(delta, pos_x, pos_y)
            self.update_theta_mu(mu, delta, feature_vector)
            self.update_theta_mean_beta(delta, beta_i, beta_mean, beta_sd, feature_vector)
            self.update_theta_sd_beta(delta, beta_i, beta_mean, beta_sd, feature_vector)

            # print("From: " +str((pos_x, pos_y))+" to: " + str((pos_x_prime, pos_y_prime)) +
            #       "\n l : " +str(l) + " beta:  " +str(beta_i) + " mu:" + str(mu) + " Reward: " +str(reward))
            self.update_state(pos_x_prime, pos_y_prime)

            # time.sleep(1)

        pos_x = self.pos_x
        pos_y = self.pos_y
        feature_vector = self.feature_vector(pos_x, pos_y)
        mu = self.sample_parameter(feature_vector)
        l = self.step_fixed
        # l = self.sample_step_length(mu)
        beta_i, beta_mean, beta_sd = self.sample_angle(feature_vector)
        pos_x_prime, pos_y_prime = self.next_state(l, beta_i)
        reward, cont = enviroment.collected_targets(pos_x_prime, pos_y_prime, radius_detection)
        self.targets_collected += cont
        delta = self.delta_T(feature_vector, reward)
        self.update_w(delta, pos_x, pos_y)
        self.update_theta_mu(mu, delta, feature_vector)
        self.update_theta_mean_beta(delta, beta_i, beta_mean, beta_sd, feature_vector)
        self.update_theta_sd_beta(delta, beta_i, beta_mean, beta_sd, feature_vector)
        # print("From: " +str((pos_x, pos_y))+" to: " + str((pos_x_prime, pos_y_prime)) +
        #       "\n l : " +str(l) + " beta:  " +str(beta_i) + " mu:" + str(mu) + " Reward: " +str(reward))
        self.update_state(pos_x_prime, pos_y_prime)
        # time.sleep(1)
        self.save_weights()


        return self.targets_collected

def initialize_weigths():

    coarse_centers = 1000 * np.arange(0, 11, 1)
    c_1 = list(itertools.product(coarse_centers, coarse_centers))
    c_2 = list(itertools.product(coarse_centers, coarse_centers))
    c_0 = [(5000, 5000)]
    c = c_0 + c_1 + c_2
    w = np.array(len(c) * [0])
    theta_mu = np.array(len(c) * [0] + len(c) * [0])
    theta_mean_beta = np.array(len(c) * [0])
    theta_sd_beta = np.array(len(c) * [-0.5])  # an approximation of 1 standard deviation
    # theta_sd_beta[0]=0.1
    # theta_sd_beta[1] = 0.1
    np.savetxt('w.csv', w, delimiter=',')
    np.savetxt('theta_mu.csv', theta_mu, delimiter=',')
    np.savetxt('theta_mean_beta.csv', theta_mean_beta, delimiter=',')
    np.savetxt('theta_sd_beta.csv', theta_sd_beta, delimiter=',')

def load_weights():

    w=np.loadtxt('w.csv', delimiter=',')
    theta_mu=np.loadtxt('theta_mu.csv', delimiter=',')
    theta_mean_beta= np.loadtxt('theta_mean_beta.csv', delimiter=',')
    theta_sd_beta= np.loadtxt('theta_sd_beta.csv', delimiter=',')

    return (w,theta_mu,theta_mean_beta,theta_sd_beta)
#Initialization values


def simulation(targets,target_location, data_weights,N_episodes,radius_coarse, L, T, radius_detection):

    #parameters of simulation
    #number of episodes
    #initialize agent
    #Each episode we save the weights

    if not data_weights:

        initialize_weigths()

    w, theta_mu, theta_mean_beta, theta_sd_beta=load_weights()

    agent=Agent(radius_coarse, L, T,radius_detection,w, theta_mu, theta_mean_beta, theta_sd_beta)
    target_list=[]

    set_weights=[1000,2000,3000,4000,5000]

    for n in range(N_episodes):

        output=print("collection " + str(agent.one_episode(L,target_location)))
        target_list.append(output)
        # agent.plot_path(targets)
        agent.reset_agent()

        if n in set_weights:

            w_name="w_"+str(n)+".csv"
            theta_mu_name="theta_mu_"+str(n)+".csv"
            theta_sd_beta_name="theta_mean_beta_"+str(n)+".csv"
            theta_mean_beta_name="theta_sd_beta_"+str(n)+".csv"

            np.savetxt(w_name, agent.w, delimiter=',')
            np.savetxt(theta_mu_name, agent.theta_mu, delimiter=',')
            np.savetxt(theta_sd_beta_name, agent.theta_mean_beta, delimiter=',')
            np.savetxt(theta_mean_beta_name, agent.theta_sd_beta, delimiter=',')

    pass

    #initialize enviroment for each episode



def plot_learning_curve():

    average=np.mean(learning_table, axis=0)
    batches=average[np.arange(0,5000,50)]

    batches=[np.mean(average[(i*50):(i*50)+50], axis=0) for i in range(100)]
    xx=[50*i for i in range(100)]
    x=np.arange(0,5000,50)
    batches[np.argmax(batches)]=3
    plt.plot(xx,batches, color='#65C7F5')
    # plt.plot(x, batches, color='#65C7F5')
    plt.xlabel("Episode")
    plt.ylabel("Average Targets collected")
    plt.title("Learning Curve")
    plt.show()

def plot_coarse_coding():
    targets = np.loadtxt('target_large.csv', delimiter=',')
    x_t=targets[:,0]
    y_t=targets[:,1]


    check_points= np.array([[2000,3000], [3000,7000], [4000,6000],[7000,4000],[9000,6000],[6000,8000],[9000,9000],[8000,2000],[1000,4000]])

    mark=np.array([[4000,6000]])

    ax = plt.gca()
    ax.set_xlim((0, 10000))
    ax.set_ylim((0, 10000))

    ax.plot(mark[:,0], mark[:,1],marker='*', color="black")
    # ax.plot(mark[:, 0], mark[:, 1], "ok", color="blue")
    ax.plot(x_t, y_t, "ok", label="input point", color="red")

    coarse_centers = 1000 * np.arange(0, 11, 1)
    centers = list(itertools.product(coarse_centers, coarse_centers))

    for c in centers:
        circle2 = plt.Circle(c, 2500, color='g', fill=False)
        ax.add_patch(circle2)
    # #     # plt.legend()
    # #     # plt.colorbar()
    # #     # plt.axis("equal")
    plt.show()
    # #
#
#
# w, theta_mu, theta_mean_beta, theta_sd_beta=load_weights()
#
# check_points = np.array([[2000, 3000], [3000, 7000], [4000, 6000], [7000, 4000], [9000, 6000], [6000, 8000], [9000, 9000], [8000, 2000],
#      [1000, 4000]])


# w, theta_mu, theta_mean_beta, theta_sd_beta=load_weights()
# n_targets = []
# radius_coarse = 750
# L = 10000
# T = 50000
# radius_detection = 25
# N = 5000
# M = 1
# robot = Agent(radius_coarse, L, T, radius_detection, w, theta_mu, theta_mean_beta, theta_sd_beta)
# robot.t=1



if __name__ == "__main__":

    #Parameter of simulation

    radius_coarse=750
    L=10000
    T=5000
    radius_detection=25
    N_episodes=1000

    data_weights=False

    w, theta_mu, theta_mean_beta, theta_sd_beta=load_weights()
    agente=Agent(radius_coarse, L, T, radius_detection, w, theta_mu, theta_mean_beta, theta_sd_beta)

    targets = np.loadtxt('target_large.csv', delimiter=',')
    target_location = set([(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])])

    simulation(targets,target_location, data_weights, N_episodes, radius_coarse, L, T, radius_detection)



