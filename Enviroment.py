import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import bernoulli

class Agent:

    def __inti__(self, pos_x, pos_y,displacement,posterior_table):
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.displacement=displacement
        self.posterior_table=posterior_table


    def movement(self, direction):

        angle=np.pi*0.25*direction

        self.pos_x=self.pos_x+ self.displacement*np.cos(angle)
        self.pos_y=self.pos_y + self.displacement*np.sin(angle)



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



