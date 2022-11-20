import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import bernoulli

class Agent:

    def __inti__(self, pos_x, pos_y,displacement,posterior_table,d):
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.displacement=displacement
        self.posterior_table=posterior_table
        self.device_factor = d


    def movement(self, direction):

        angle=np.pi*0.25*direction

        self.pos_x=self.pos_x+ self.displacement*np.cos(angle)
        self.pos_y=self.pos_y + self.displacement*np.sin(angle)

    def update_posterior_table(self, rewards_table):

        n=self.posterior_table.shape[0]
        temp_likelihood=self.posterior_table[:,2]
        relative_distance=[distance.euclidean((rewards_table[0,0],rewards_table[0,1]),(self.posterior_table[i,0],self.posterior_table[i,1])) for i in range(n)]

        n_signal_ones=sum(rewards_table[:,2])
        n_signal_zeros=rewards_table.shape[0]-n_signal_ones

        ones_signal=np.array([(relative_distance[i]/self.device_factor)*n_signal_ones for i in range(n) ])
        zeros_signal=np.array([n_signal_zeros*np.log(1-np.exp(relative_distance[i]/self.device_factor)) for i in range(n)])

        update_likelihood=ones_signal+zeros_signal+temp_likelihood

        self.posterior_table[:, 2]=update_likelihood






class Enviroment:

    def __init__(self,L,target_x,target_y,d,number_signals,epsilon):

        self.lenght=L
        self.target_x=target_x
        self.target_y=target_y
        self.device_factor=d
        self.number_signals
        self.epsilon=epsilon


    def generate_signal(self, pos_x, pos_y):

        l_between_point=distance.euclidean((pos_x,pos_y),(self.target_x,self.target_y))
        likelihood= np.exp(-l_between_point/self.device_factor)

        return bernoulli.rvs(likelihood, size=self.number_signals)



    def target_found(self, pos_x, pos_y):

        if (distance.euclidean((pos_x, pos_y),(self.target_x, self.target_y))<=self.epsilon):

            return True

        else:
            return False



