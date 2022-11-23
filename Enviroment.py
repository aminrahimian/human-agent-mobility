import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import bernoulli
import itertools
import pickle
from numpy.random import choice
from scipy.stats import norm


class Mobile_unit:

    def __init__(self, pos_x, pos_y,displacement,posterior_table,d,time_steps):
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.displacement=displacement
        self.posterior_table=posterior_table
        self.device_factor = d
        self.time_steps=time_steps


    def movement(self, direction):

        # angle=(np.pi)*0.25*(float(direction))

        self.pos_x=self.pos_x+ self.displacement*np.cos((np.pi)*0.25*(float(direction)))
        self.pos_y=self.pos_y + self.displacement*np.sin((np.pi)*0.25*(float(direction)))

    def soft_exponential(self,a):
        return 1.1**a


    def update_posterior_table(self, rewards_table):

        n=self.posterior_table.shape[0]
        temp_likelihood=self.posterior_table[:,2]
        relative_distance=[-distance.euclidean((rewards_table[0,0],rewards_table[0,1]),(self.posterior_table[i,0],self.posterior_table[i,1])) for i in range(n)]

        n_signal_ones=sum(rewards_table[:,2])
        n_signal_zeros=rewards_table.shape[0]-n_signal_ones

        ones_signal=np.array([(relative_distance[i]/self.device_factor)*n_signal_ones for i in range(n) ])
        zeros_signal=np.array([n_signal_zeros*np.log(1-np.exp(relative_distance[i]/self.device_factor)) for i in range(n)])

        update_likelihood=ones_signal+zeros_signal+temp_likelihood

        self.posterior_table[:, 2]=update_likelihood


    def sample_from_posterior(self,n):

        probability_distribution = self.soft_exponential(self.posterior_table[:,2]) / np.sum(self.soft_exponential(self.posterior_table[:,2]))
        list_of_candidates=list(range(self.posterior_table.shape[0]))

        draw = choice(list_of_candidates, n, p=probability_distribution)

        average_x=sum(self.posterior_table[draw,0])/n
        average_y=sum(self.posterior_table[draw,1])/n

        return [average_x,average_y]

class Enviroment:

    def __init__(self,L,target_x,target_y,phi,number_signals,epsilon):

        self.lenght=L
        self.target_x=target_x
        self.target_y=target_y
        self.device_factor=phi
        self.number_signals=number_signals
        self.epsilon=epsilon


    def generate_signal(self, pos_x, pos_y):

        table=np.zeros((self.number_signals,3))
        l_between_point=distance.euclidean((pos_x,pos_y),(self.target_x,self.target_y))
        likelihood= np.exp(-l_between_point/self.device_factor)
        table[:,0]=np.array([pos_x]*self.number_signals)
        table[:,1] = np.array([pos_y] * self.number_signals)
        table[:,2]=bernoulli.rvs(likelihood, size=self.number_signals)

        return table


    def target_found(self, pos_x, pos_y):

        if (distance.euclidean((pos_x, pos_y),(self.target_x, self.target_y))<=self.epsilon):

            return True

        else:
            return False

def create_target_instances():

    targetx = np.random.uniform(0.05, 0.95, 100)
    targety = np.random.uniform(0.05, 0.95, 100)

    target_instances = np.array([targetx, targety]).T

    with open('target_instances.pkl', 'wb') as f:
        pickle.dump(target_instances, f)

def load_instances(L):

    with open('target_instances.pkl', 'rb') as f:

                target_instances = pickle.load(f)

    target_instances[:,0]=target_instances[:,0]*L
    target_instances[:,1] =target_instances[:,1]*L

    return target_instances

class Search_Methods:

    def bayes_n_sample(self,L,n):

        target_instances = load_instances(L)
        Time_record = []
        prior_theta1 = L * 0.5
        prior_theta2 = L * 0.5
        sd_prior = L

        for t in range(100):

            area1 = Enviroment(L, target_instances[t, 0], target_instances[t, 1], 200, 50, 100)
            linespacing = int(np.ceil((area1.lenght * 1) / (area1.epsilon * np.sqrt(2))))
            mc_theta1 = np.linspace(1, area1.lenght, linespacing, endpoint=True)
            mc_theta2 = np.linspace(1, area1.lenght, linespacing, endpoint=True)
            all_coordinates = list(itertools.product(mc_theta1, mc_theta2))
            test_posteriors = np.zeros((len(all_coordinates), 3))
            test_posteriors[:, 0] = [all_coordinates[i][0] for i in range(len(all_coordinates))]
            test_posteriors[:, 1] = [all_coordinates[i][1] for i in range(len(all_coordinates))]
            test_posteriors[:, 2] = [np.log(
                norm.pdf(all_coordinates[i][0], loc=prior_theta1, scale=sd_prior) + norm.pdf(all_coordinates[i][1],
                                                                                             loc=prior_theta1,
                                                                                             scale=sd_prior)) for i in
                                     range(len(all_coordinates))]

            agente = Mobile_unit(0, 0, 140, test_posteriors, 200, 0)

            while not area1.target_found(agente.pos_x, agente.pos_y):

                # print if we havent found the target yet
                # print the position
                # print the direction
                # print(" Target? :"+  str(area1.target_found(agente.pos_x,agente.pos_y)))

                agente.time_steps += 1
                x1 = agente.pos_x
                x2 = agente.pos_y

                # print("pos x : " +str(x1))
                # print("pos y : " + str(x2))
                # print("Target " + str(area1.target_found(x1,x2)))

                lgth = agente.displacement

                comparison_table = np.array([[x1 + lgth, x2, 0],
                                             [x1 + lgth * np.cos(np.pi * 0.25), x2 + lgth * np.sin(np.pi * 0.25), 0],
                                             [x1, x2 + lgth, 0],
                                             [x1 - lgth * np.cos(np.pi * 0.25), x2 + lgth * np.cos(np.pi * 0.25), 0],
                                             [x1 - lgth, x2, 0],
                                             [x1 - lgth * np.cos(np.pi * 0.25), x2 - lgth * np.cos(np.pi * 0.25), 0],
                                             [x1, x2 - lgth, 0],
                                             [x1 + lgth * np.cos(np.pi * 0.25), x2 - lgth * np.cos(np.pi * 0.25), 0]])

                local_opt = agente.sample_from_posterior(n)
                local_opt_x = local_opt[0]
                local_opt_y = local_opt[1]

                # diff=agente.posterior_table[np.argmax(agente.posterior_table[:, 2]), 2]- agente.posterior_table[167,2]

                # print("Guess x : " + str(local_opt_x))
                # print("Guess y  : " + str(local_opt_y))
                # print("Local optimal: " +str(agente.posterior_table[np.argmax(agente.posterior_table[:, 2]), 2]))

                # print(" Difference Likelihoods: " +str(diff))

                for i in range(8):
                    comparison_table[i, 2] = distance.euclidean((comparison_table[i, 0], comparison_table[i, 1]),
                                                                (local_opt_x, local_opt_y))

                which_dir = np.argmin(comparison_table[:, 2])

                agente.movement(which_dir)

                rewards = area1.generate_signal(x1, x2)

                agente.update_posterior_table(rewards)

                if agente.time_steps >= 2000:
                    break

            Time_record.append(agente.time_steps)
            a = np.array(Time_record)
            file_name = "LMAB" + str(L) + ".csv"
            np.savetxt(file_name, a, delimiter=",")

        print(np.mean(Time_record))

    def thorough_search(self,L):

        target_instances = load_instances(L)
        Time_record=[]

        for t in range(100):

            area1 = Enviroment(L, target_instances[t, 0], target_instances[t, 1], 200, 50, 100)
            linespacing = int(np.ceil((area1.lenght * 1) / (area1.epsilon * np.sqrt(2))))
            mc_theta1 = np.linspace(1, area1.lenght, linespacing, endpoint=True)
            mc_theta2 = np.linspace(1, area1.lenght, linespacing, endpoint=True)
            all_coordinates = list(itertools.product(mc_theta1, mc_theta2))
            test_posteriors = np.zeros((len(all_coordinates), 3))
            test_posteriors[:, 0] = [all_coordinates[i][0] for i in range(len(all_coordinates))]
            test_posteriors[:, 1] = [all_coordinates[i][1] for i in range(len(all_coordinates))]
            test_posteriors[:, 2] = [np.log(norm.pdf(all_coordinates[i][0], loc=prior_theta1, scale=sd_prior) + norm.pdf(all_coordinates[i][1],
                                                                                             loc=prior_theta1,
                                                                                             scale=sd_prior)) for i in
                                     range(len(all_coordinates))]
            agente = Mobile_unit(0, 0, 140, test_posteriors, 200, 0)

            directions = np.array([])
            grid_size = int(np.ceil(area1.lenght / agente.displacement))
            step_size = agente.displacement

            # Generate deterministic movement policy

            for i in range(grid_size):

                if i % 2 == 0:

                    forward = np.array([0] * grid_size)
                    forward[-1] = 2
                    directions = np.concatenate((directions, forward))


                else:

                    backward = np.array([4] * grid_size)
                    backward[-1] = 2
                    directions = np.concatenate((directions, backward))

            for i in directions:
                    # print if we havent found the target yet
                    # print the position
                    # print the direction
                    print(" Target? :" + str(area1.target_found(agente.pos_x, agente.pos_y)))

                    agente.time_steps += 1
                    x1 = agente.pos_x
                    x2 = agente.pos_y

                    print("pos x : " + str(x1))
                    print("pos y : " + str(x2))

                    agente.movement(i)

                    if area1.target_found(agente.pos_x, agente.pos_y):

                        break

                    if agente.time_steps >= 1000:
                        break


            Time_record.append(agente.time_steps)
            a=np.array(Time_record)
            file_name="Through_search_"+str(L)+".csv"
            np.savetxt(file_name, a, delimiter=",")


    def short_path(self):
        pass




if __name__=="__main__":


    L=1000      # Lenght of the side of the area

    # Change the method according to the search algorithm
    # Output a cvs file with the time steps for 10 instances.

    Search_Methods.thorough_search(L)



