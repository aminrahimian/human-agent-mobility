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

    def __init__(self,T,load_weights):

        self.L= 10000
        self.pos_x=self.L/2
        self.pos_y=self.L/2
        grid_line =  np.arange(0, 10050, 50)
        self.center_lay1 = list(itertools.product(grid_line, grid_line))
        self.center_lay2 = list(itertools.product(grid_line, grid_line))
        self.center_init = [(self.L/2, self.L/2)]
        self.centers = self.center_init + self.center_lay1
        self.alpha=0.1

        if load_weights:

            self.weights = np.loadtxt('weights_SARSA.csv', delimiter=',')

        else:

            self.weights = np.array([0] + len(self.center_lay1) * [0] +
                                    [-1] + len(self.center_lay1) * [-1])

        # when no data


        # when data is available

        #

        # when simulation

        # self.t=0

        #when ploting

        self.t=0
        self.T=T
        self.path_x = []
        self.path_y = []
        self.epsilon=0.1
        # define these parameters carefully
        self.radius_coarse=35
        self.radius_detection=25
        self.terminal_state=False

    def feature_vector(self, pos_x, pos_y, action):

        zeros = [0] + len(self.center_lay1) * [0]

        if self.t==0:

            # print(" time step  initial ------------")


            feature_vector = [1] + len(self.center_lay1) * [0]

            if action==2:

                feature_vector=np.array(feature_vector+zeros)

            else:

                feature_vector=np.array(zeros+ feature_vector)

        else:

            # print(" time step diff from initial")

            actual_pos = len(self.center_lay1) * [(pos_x, pos_y)]
            iter = len(self.center_lay1)
            distances = np.array([distance.euclidean(self.center_lay1[i], actual_pos[i]) for i in range(iter)])
            inside_circle_1 = (distances <= self.radius_coarse)
            inside_circle_1 = list(inside_circle_1.astype(int))

            inside_circle = [0] + inside_circle_1

            if action==2:

                feature_vector = np.array(inside_circle + zeros)

            else:

                feature_vector = np.array(zeros + inside_circle)

        return feature_vector

    def sample_step_length(self, mu):

        alpha = mu - 1
        V = np.random.uniform(-np.pi * 0.5, 0.5 * np.pi, 1)[0]
        W = expon.rvs(size=1)[0]
        cop1 = (np.sin(alpha * V)) / ((np.cos(V)) ** (1 / alpha))
        cop2 = ((np.cos((1 - alpha) * V)) / W) ** (((1 - alpha) / alpha))
        l = cop1 * cop2

        ## set the

        if l <= 0:

            l = 10*min(500, -l)
            step_l = max(25, l)

        else:

            l = 10*min(500, l)
            step_l = max(25,l)

        # print("Taking this value:  " + str(elle))

        # angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # # print("angle vel :" + str(angle))
        # vel_x = 10 * elle * np.cos(angle)
        # vel_y = 10 * elle * np.sin(angle)

        return step_l

    def epsilon_greedy_policy(self, pos_x, pos_y):

        """
        Return an action [a] considering the q(s,a) chosing max{q(s,a)}
        with probability 1-epsilon
        :param pos_x: position x
        :param pos_y: position y
        :return: parameter mu of levy distribution
        """
        actions=[2,3]

        feature_mu2=self.feature_vector(pos_x, pos_y,actions[0])
        feature_mu3=self.feature_vector(pos_x, pos_y,actions[1])

        q_values=[np.dot(self.weights,feature_mu2), np.dot(self.weights,feature_mu3)]

        print("q values :" +str(q_values))
        sample_uniform=np.random.uniform(0, 1,1)[0]

        if sample_uniform>=self.epsilon:

            # print("Action taking by MAX  ::::   ::::  ")

            index_max=np.argmax(q_values)
            mu=actions[index_max]

        else:

            # print("Action taking by SAMPLING  ::::   ::::  ")
            mu=np.random.choice(np.array(actions))

        return mu

    def q_val_ratio(self, pos_x, pos_y):

        # feature_mu2 = self.feature_vector(pos_x, pos_y,2)
        feature_mu3 = self.feature_vector(pos_x, pos_y, 3)

        # q_u_2=np.dot(self.weights, feature_mu2)
        q_u_3=np.dot(self.weights, feature_mu3)

        # if q_u_2>=q_u_3:
        #
        #     val=2
        #
        # else:
        #
        #     val=3

        return q_u_3

    def next_position(self, mu, pos_x, pos_y):

        lenght_step=self.sample_step_length(mu)
        # print(" length step : " + str(lenght_step))
        angle = np.random.uniform(0, 2 * np.pi, 1)[0]
        # print("angle vel :" + str(angle))
        vel_x =  lenght_step * np.cos(angle)
        vel_y =  lenght_step * np.sin(angle)

        pos_x_prime = pos_x + vel_x
        pos_y_prime = pos_y + vel_y

        if pos_x_prime > self.L:
            pos_x_prime = 2 * self.L - pos_x_prime

        if pos_x_prime < 0:
            self.pos_x_prime = -1 * pos_x_prime

        if pos_y_prime > self.L:
            pos_y_prime = 2 * self.L - pos_y_prime

        if pos_y_prime < 0:
            pos_y_prime = -1 * pos_y_prime

        return (pos_x_prime, pos_y_prime)

    def update_weights(self, pos_x, pos_y,pos_x_prime, pos_y_prime, current_action,
                       next_action, reward):

        if not self.terminal_state:

            next_q_hat = np.dot(self.feature_vector(pos_x_prime, pos_y_prime,next_action), self.weights)
            current_q_hat = np.dot(self.feature_vector(pos_x, pos_y,current_action), self.weights)
            nabla_q_hat=self.feature_vector(pos_x, pos_y,current_action)
            delta=self.alpha*(reward+next_q_hat-current_q_hat)
            self.weights=self.weights+delta*nabla_q_hat

        else:

            current_q_hat = np.dot(self.feature_vector(pos_x, pos_y, current_action), self.weights)
            nabla_q_hat = self.feature_vector(pos_x, pos_y, current_action)
            delta = self.alpha * (reward  - current_q_hat)
            self.weights = self.weights + delta * nabla_q_hat

    def update_agent_state(self, pos_x_prime, pos_y_prime):

        self.path_x.append(self.pos_x)
        self.path_y.append(self.pos_y)
        self.pos_x=pos_x_prime
        self.pos_y=pos_y_prime
        self.t+=1

        ## check this one

        if self.t==self.T:
            
            self.terminal_state=True

    def reset_agent(self):

        self.pos_x = self.L / 2
        self.pos_y = self.L / 2
        self.t = 0
        self.path_x = []
        self.path_y = []
        self.terminal_state = False

    def plot_path(self, targets_coord):

        targets_x = []
        targets_y = []
        targets = list(targets_coord)

        for r in targets:
            targets_x.append(r[0])
            targets_y.append(r[1])

        plt.plot(self.path_x, self.path_y, '--o', color='#65C7F5')
        plt.plot(targets_x, targets_y, 'ko')
        plt.xlim([0, 10000])
        plt.ylim([0, 10000])
        plt.show()

    def episodic_sarsa(self, iterations):

        for n in range(iterations):

            targets = np.loadtxt('target_large.csv', delimiter=',')
            targets_coord = set([(targets[i, 0], targets[i, 1]) for i in range(targets.shape[0])])
            e1 = Enviroment(targets_coord)

            for p in range(self.T):

                print(" --------       step " + str(p) + "- iter " + str(n) + "           -----")

                pos_x = agent.pos_x
                pos_y = agent.pos_y
                mu = agent.epsilon_greedy_policy(pos_x, pos_y)
                current_action = mu
                print("Paramter action " + str(current_action))
                pos_x_prime, pos_y_prime = agent.next_position(mu, pos_x, pos_y)
                reward = e1.collected_targets(pos_x_prime, pos_y_prime, agent.radius_detection)
                next_action = agent.epsilon_greedy_policy(pos_x_prime, pos_y_prime)
                print(" Agent From :  " + str((pos_x, pos_y)) + " to : " + str((pos_x_prime, pos_y_prime)))
                print("Reward:  " + str(reward))
                agent.update_weights(pos_x, pos_y, pos_x_prime, pos_y_prime, current_action, next_action, reward)
                agent.update_agent_state(pos_x_prime, pos_y_prime)
                print("Terminal state " + str(agent.terminal_state))
                # time.sleep(2)

            self.plot_path(targets_coord)

            np.savetxt("weights_SARSA.csv", agent.weights, delimiter=",")

            if not agent.terminal_state:

                break

            else:

                agent.reset_agent()

    def plot_heat_map(self):

        targets = np.loadtxt('target_large.csv', delimiter=',')
        samples = 1000
        rng = np.random.default_rng()
        x = 10000 * np.random.uniform(0, 1, samples)
        y = 10000 * np.random.uniform(0, 1, samples)
        z = [self.q_val_ratio(x[i], y[i]) for i in range(samples)]
        X = np.linspace(min(x), max(x))
        Y = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        interp = NearestNDInterpolator(list(zip(x, y)), z)
        Z = interp(X, Y)
        plt.pcolormesh(X, Y, Z, shading='auto')
        plt.plot(targets[:, 0], targets[:, 1], 'ok')
        # plt.legend()
        plt.colorbar()
        plt.axis("equal")
        plt.show()


# e1.plot_target()

if __name__ == "__main__":

    T=1000
    agent=Agent(T)
    iterations=10

    agent.episodic_sarsa(iterations)
    agent.plot_heat_map()


    ## TO DO

