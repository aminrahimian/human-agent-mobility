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

    def collected_targets(self, agent):

        """ Return the number of targets given the historical
        position of the agent
        """
        p_x = agent.pos_x
        p_y = agent.pos_y
        agent_position=(p_x,p_y)

        cont=0
        initial_range=len(self.target_location)
        removal=set({})
        list_target_location=list(self.target_location)

        for i in range(initial_range):

            if distance.euclidean(agent_position, list_target_location[i])<= agent.radius_detection:

                removal.add(list_target_location[i])
                cont+=1

            else:

                pass

        self.target_location=self.target_location.difference(removal)

        if cont > 0:
            print("Total targets : " + str(cont))

        return cont

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
        self.theta_mean_alpha = np.array(len(self.c) * [0])
        self.theta_sd_alpha = np.array(len(self.c) * [1])  # an approximation of 1 standard deviation

        #---------------------------------------------------------------

        self.radius_coarse = radius_coarse
        self.radius_detection=radius_detection
        self.cum_targets = 0
        self.alpha = 1e-5
        self.gamma = 1
        self.time_step = 0
        self.parameter=2
        self.A=[]
        self.episode=0
        self.path_x=[self.pos_x]
        self.path_y=[self.pos_y]
        self.targets_collected=[]

    def feature_vector(self, pos_x, pos_y):

        if self.episode==0:

            inside_circle = np.array([1] + len(self.center_lay1) * [0] + len(self.center_lay2) * [0])
            # print(" init state")

        elif (self.episode>0) & (self.episode+1<=self.T):

            # print(">1")
            actual_pos = len(self.c_1) * [(pos_x, pos_y)]
            iter = len(self.center_lay1)
            distances = np.array([distance.euclidean(self.center_lay1[i], actual_pos[i]) for i in range(iter)])
            inside_circle_1 = (distances <= self.radius_coarse)
            inside_circle_1 = list(inside_circle_1.astype(int))

            inside_circle=[0]+ inside_circle_1+ len(self.center_lay1) * [0]


        else:

            # print("Final grid ")
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

    def sample_parameter(self, pos_x, pos_y):

        #calculate h(s,a theta)

        partial_x_sa=list(self.feature_vector(pos_x, pos_y))
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

    def sample_action(self):

        self.A.append(self.sample_parameter())
        mu=self.parameter-1

        l=pareto.rvs(mu, scale=250,size=1)[0]
        angle=np.random.uniform(0,2*np.pi,1)[0]
        # print("angle vel :" + str(angle))
        vel_x = l*np.cos(angle)
        vel_y = l*np.sin(angle)

        return (vel_x, vel_y)

    def sample_action_levy(self):

        self.A.append(self.sample_parameter())
        alpha= self.parameter -1
        V = np.random.uniform(-np.pi*0.5, 0.5*np.pi, 1)[0]
        W = expon.rvs(size=1)[0]
        cop1=(np.sin(alpha*V))/((np.cos(V))**(1/alpha))
        cop2=((np.cos((1-alpha)*V))/W)**(((1-alpha)/alpha))
        elle=cop1*cop2

        if elle<=0:

            elle=min(500,-elle)

        else:

            elle=min(500, elle)

        # print("Taking this value:  " + str(elle))

        angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # print("angle vel :" + str(angle))
        vel_x = 10*elle * np.cos(angle)
        vel_y = 10*elle * np.sin(angle)

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

    def nabla_pi_sa_II(self, pos_x, pos_y, action):

        partial_x_sa = list(self.feature_vector(pos_x, pos_y))
        zeros_x_sa = len(partial_x_sa) * [0]
        x_sa_mu_2 = partial_x_sa + zeros_x_sa
        x_sa_mu_3 = zeros_x_sa + partial_x_sa

        h_mu_2 = np.dot(self.theta, x_sa_mu_2)
        h_mu_3 = np.dot(self.theta, x_sa_mu_3)

        p_mu_2 = np.exp(h_mu_2) / (np.exp(h_mu_2) + np.exp(h_mu_3))
        p_mu_3 = 1 - p_mu_2

        # print("p_mu_2, p_mu_3: " + str((p_mu_2, p_mu_3)))

        cum_soft_max = p_mu_2 * np.array(x_sa_mu_2) + \
                       p_mu_3 * np.array(x_sa_mu_3)

        if action == 2:

            nabla = x_sa_mu_2 - cum_soft_max

        elif action==3:

            nabla = x_sa_mu_3 - cum_soft_max

        return nabla

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
        number_targets = enviroment.collected_targets(self)
        self.targets_collected.append(number_targets)
        reward=self.targets_collected[-1]

        if reward==0:

            reward=-10

        else:

            reward=100

        # print("current position: " +str((a1.pos_x, a1.pos_y)))
        # print("next  position: " + str((a1.pos_x_prime, a1.pos_y_prime)))
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

        # number_targets=env.collected_targets(self)
        #
        # if number_targets>0:
        #
        #     print("Add to target collected list : " +str(number_targets) +
        #           "at :" +str((self.pos_x,self.pos_y)))
        #
        # self.targets_collected.append(number_targets)

        return delta

    def generate_one_step(self,action, enviroment):

        self.update_next_state(action)

        # print("current position: " +str((a1.pos_x, a1.pos_y)))
        # print("next  position: " + str((a1.pos_x_prime, a1.pos_y_prime)))

        self.update_state()

        number_targets = enviroment.collected_targets(self)
        self.targets_collected.append(number_targets)


        return 1

    def monte_carlo_update(self):

        mc_path_x = self.path_x[0:self.episodes]
        mc_path_y = self.path_y[0:self.episodes]

        R=[]

        for i in self.targets_collected:

            if i>0:

                R.append(1000)

            else:

                R.append(-10)

        # time.sleep(3)


        for i in range(len(R)):

            self.episode=i
            G=np.sum(R[0:i+1])
            # print("G value  " + str(G))
            self.theta = self.theta + \
                         self.alpha * G * self.nabla_pi_sa_II(mc_path_x[i],
                                                              mc_path_y[i],self.A[i])

    def reset_agent(self):

        self.pos_x = L / 2
        self.pos_y = L / 2
        self.pos_x_prime = L / 2
        self.pos_y_prime = L / 2
        self.episode=0
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

        plt.plot(a1.path_x, a1.path_y, '--o', color='#65C7F5')
        plt.plot(targets[:, 0], targets[:, 1], 'ko')
        plt.show()


#Initialization values

if __name__ == "__main__":

    n_targets=[]

    radius_coarse=1500
    L=10000
    T=50
    radius_detection=25
    freq_sampling=1
    episodes=1000

    # Define Agent object

    a1=Agent(radius_coarse,L,T,radius_detection, freq_sampling)

    #Learning block

    scenarios=50

    for j in range(scenarios):

        targets = np.loadtxt('target_large.csv', delimiter=',')
        target_location = set([(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])])

        env = Enviroment(L, target_location)
        # Loop for episodes
        print("learning with  episode : " +str(j))

        for i in range(episodes):

            # Loop for one complete episode

            # print("==============================")
            # print("Episode: " + str(a1.episode))
            action = a1.sample_action_levy()
            # print("action " +str (action))
            ## use this if one-step actor critic

            # a1.updates_weigts( action, env)

            a1.generate_one_step(action, env)
            # time.sleep(1)

        print("for episode : " +str(j) + " A : " +str(a1.A))

        n_targets.append(np.sum(a1.targets_collected))

        a1.monte_carlo_update()

        if j==(scenarios-1):

            break

        a1.reset_agent()



    a1.plot_path(targets)
    a1.targets_collected[-1]


    np.savetxt("n_targets_05_2k.csv",n_targets , delimiter=",")


vector_x= np.linspace(start=0, stop=10000,endpoint=True ,num=2000)
vector_y= np.linspace(start=0, stop=10000,endpoint=True ,num=2000)

a1.feature_vector(vector_x[0], vector_y[0])

a1.weigths_vector

# df = pd.DataFrame({'x':vector_x ,
#                    'y': vector_y,
#                    'z': v_hat})
#
# df.to_csv("interpolation.csv",index=False)

# but those can be array_like structures too, in which case the
# result will be a matrix representing the values in the grid
# specified by those arguments

a1.episode=4

targets = np.loadtxt('target_large.csv', delimiter=',')
x_t=targets[:,0]
y_t=targets[:,1]

rng = np.random.default_rng()
x = 10000 *np.random.uniform(0,1,2000)
y = 10000 *np.random.uniform(0,1,2000)
z=[a1.probability_actions(vector_x[i], vector_y[i]) for i in range(2000)]
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = NearestNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto')
plt.plot(x_t, y_t, "ok", label="input point", color="red")
# plt.legend()
# plt.colorbar()
# plt.axis("equal")
plt.show()


R=a1.targets_collected
a1.theta

p_mu_2=0.1
random.choices([2,3], weights=[p_mu_2, 1-p_mu_2], k=1)[0]