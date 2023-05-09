import pandas as pd
import numpy as np
from scipy.stats import norm
import itertools
from scipy.spatial import distance


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

L=2000
n_targets=2000



class Enviroment:

    def __init__(self,L,n_targets):

        self.dim=L
        self.distribution_mean_x=L/2
        self.distribution_mean_y=L/2
        self.distribution_sd_x=200
        self.distribution_sd_y=200

        tx = norm.rvs(loc=self.distribution_mean_x, scale=self.distribution_sd_x,size=n_targets)
        ty = norm.rvs(loc=self.distribution_mean_y, scale=self.distribution_sd_y, size=n_targets)

        tx=tx.astype(int)
        ty=ty.astype(int)


        self.targets=set([(tx[i],ty[i]) for i in range(n_targets)])


    def reward_one_step(self,sweep_positions):

        to_delete=[]
        for i in itertools.product(list(self.targets), list(sweep_positions)):
            comp = (distance.euclidean(i[0], i[1]))
            if comp <= 10:
                to_delete.append(i[0])

        to_delete=set(to_delete)

        reward=len(to_delete)

        self.targets=self.targets.difference(to_delete)

        return reward

class Agent:


    def __init__(self):

        self.posx=0
        self.posy=0
        self.H=100
        self.alpha=1e-10

        actions = [0, np.pi / 2, np.pi, 1.5 * np.pi]
        velocities=[10,20,50]

        temp_vector=[]

        for i in itertools.product(actions,velocities):
            temp_vector.append(i)

        l1=list(range(len(temp_vector)))
        d1=dict(zip(l1,temp_vector))
        self.angles_vel=d1

        self.vector_weights=(1e-12)*np.ones((len(temp_vector),5))


    def which_action(self):

        # add randomization
        epsilon=0.1

        if np.random.uniform(0,1)<epsilon:

            return np.random.randint(0,12, size=1)[0]

        else:

            x_1=self.H*np.floor(self.posx/self.H)+ self.H*0.5
            x_2=self.H*np.floor(self.posy/self.H)+ self.H*0.5

            vector_x=np.array([x_1**2, x_1, x_2**2, x_2, 1])

            return np.where(max(self.vector_weights@vector_x)==self.vector_weights@vector_x)[0][0]

    def sample_location(self,action):

        sample_location_x=[]
        sample_location_y= []

        for i in range(25):

            add_x=self.posx + (i / 25)*self.angles_vel[action][1] * np.cos(self.angles_vel[action][0])
            add_y=self.posy + (i / 25) * self.angles_vel[action][1] * np.sin(self.angles_vel[action][0])

            if add_x>L:
                add_x=2*L -add_x

            if add_x<0:

                add_x=-1*add_x

            if add_y>L:

                add_y=2*L -add_y

            if add_y<0:

                add_y=-1*add_y

            sample_location_x.append(add_x)
            sample_location_y.append(add_y)

        sample_locations = [(sample_location_x[i], sample_location_y[i]) for i in range(25)]

        return sample_locations


    def new_positions(self,action):

        new_x = self.posx + self.angles_vel[action][1] * np.cos(self.angles_vel[action][0])

        if new_x>L:
            new_x= 2*L - new_x


        if new_x<0:
            new_x=-1*new_x

        new_y = self.posy + self.angles_vel[action][1] * np.sin(self.angles_vel[action][0])

        if new_y>L:

            new_y=2*L-new_y

        if new_y<0:
            new_y=-1*new_y

        return (new_x,new_y)


    def update_weights(self,enviroment,action):


        sample_locations=self.sample_location(action)
        row=action

        x_1 = self.H * np.floor(self.posx / self.H) + self.H * 0.5
        x_2 = self.H * np.floor(self.posy / self.H) + self.H * 0.5

        vector_x = np.array([x_1 ** 2, x_1, x_2 ** 2, x_2, 1])

        x_1_prime = self.H * np.floor((self.new_positions(action)[0]) / self.H) + self.H * 0.5
        x_2_prime = self.H * np.floor((self.new_positions(action)[1])/ self.H) + self.H * 0.5

        vector_x_prime = np.array([x_1_prime ** 2, x_1_prime, x_2_prime ** 2, x_2_prime, 1])

        delta= self.alpha*(enviroment.reward_one_step(sample_locations)+ np.dot(self.vector_weights[action,:],vector_x_prime)- np.dot(self.vector_weights[action,:],vector_x))*vector_x
        self.vector_weights[action,:]= self.vector_weights[action,:]+delta

        self.posx=self.new_positions(action)[0]
        self.posy = self.new_positions(action)[1]

        return delta



    def update_weights_end(self,enviroment,action):
        sample_locations_x = np.array([1, 2, 4])
        sample_locations_y = np.array([5, 6, 6])

        sample_locations = np.array([5, 6, 6])

        row = action

        x_1 = self.H * np.floor(self.posx / self.H) + self.H * 0.5
        x_2 = self.H * np.floor(self.posy / self.H) + self.H * 0.5

        vector_x = np.array([x_1 ** 2, x_1, x_2 ** 2, x_2, 1])

        x_1_prime = self.H * np.floor((self.new_positions(action)[0]) / self.H) + self.H * 0.5
        x_2_prime = self.H * np.floor((self.new_positions(action)[1]) / self.H) + self.H * 0.5

        vector_x_prime = np.array([x_1_prime ** 2, x_1_prime, x_2_prime ** 2, x_2_prime, 1])

        delta = self.alpha * (enviroment.reward_one_step(sample_locations) + np.dot(self.vector_weights[action, :],
                                                                                    vector_x_prime)) * vector_x
        self.vector_weights[action, :] = self.vector_weights[action, :] + delta

        self.posx = self.new_positions(action)[0]
        self.posy = self.new_positions(action)[1]



envi1=Enviroment(L,n_targets)
ag1=Agent()


for i in range(40):

    action_t=ag1.which_action()
    print(action_t)
    print(ag1.update_weights(envi1,action_t))


ag1.vector_weights

ag1.posy