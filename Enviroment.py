import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import bernoulli


class Enviroment:

    def __init__(self,L,target_x,target_y,d):

        self.lenght=L
        self.target_x=target_x
        self.target_y=target_y
        self.device_factor=d


    def generate_signal(self, pos_x, pos_y):

        l_between_point=distance.euclidean((pos_x,pos_y),(self.target_x,self.target_y))
        likelihood= np.exp(-l_between_point/self.device_factor)

        return bernoulli.rvs(likelihood, size=1)[0]
