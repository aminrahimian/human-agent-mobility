import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import bernoulli
import itertools

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




class Enviroment:

    def __init__(self,L,target_x,target_y,d,number_signals,epsilon):

        self.lenght=L
        self.target_x=target_x
        self.target_y=target_y
        self.device_factor=d
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




### Test instances

agente=Mobile_unit(0, 0,168,1,100,0)
area1=Enviroment(2000,1750,1750,200,50,100)
linespacing = int(np.ceil((area1.lenght + area1.lenght * 0.05) * np.sqrt(2) / area1.epsilon))
mc_theta1 = np.linspace(10, area1.lenght, linespacing, endpoint=True)
mc_theta2 = np.linspace(10, area1.lenght, linespacing, endpoint=True)
all_coordinates = list(itertools.product(mc_theta1, mc_theta2))
test_posteriors=np.zeros((len(all_coordinates),3))
test_posteriors[:,0]=[all_coordinates[i][0] for i in range(len(all_coordinates))]
test_posteriors[:,1]=[all_coordinates[i][1] for i in range(len(all_coordinates))]


agente=Mobile_unit(0, 0,168,test_posteriors,100,0)


for i in range(200):

    # print if we havent found the target yet
    # print the position
    # print the direction


    x1 = agente.pos_x
    x2 = agente.pos_y

    print("pos x : " +str(x1))
    print("pos y : " + str(x2))
    print("Target " + str(area1.target_found(x1,x2)))

    rewards=area1.generate_signal(x1,x2)

    agente.update_posterior_table(rewards)

    lgth = agente.displacement

    comparison_table = np.array([[x1 + lgth, x2, 0],
                                 [x1 + lgth * np.cos(np.pi * 0.25), x2 + lgth * np.sin(np.pi * 0.25), 0],
                                 [x1, x2 + lgth, 0],
                                 [x1 - lgth * np.cos(np.pi * 0.25), x2 + lgth * np.cos(np.pi * 0.25), 0],
                                 [x1 - lgth, x2, 0],
                                 [x1 - lgth * np.cos(np.pi * 0.25), x2 - lgth * np.cos(np.pi * 0.25), 0],
                                 [x1, x2 - lgth, 0],
                                 [x1 + lgth * np.cos(np.pi * 0.25), x2 - lgth * np.cos(np.pi * 0.25), 0]])

    local_opt_x = agente.posterior_table[np.argmax(agente.posterior_table[:, 2]), 0]
    local_opt_y = agente.posterior_table[np.argmax(agente.posterior_table[:, 2]), 1]

    for i in range(8):
        comparison_table[i, 2] = distance.euclidean((comparison_table[i, 0], comparison_table[i, 1]),
                                                    (local_opt_x, local_opt_y))

    which_dir = np.argmin(comparison_table[:, 2])

    agente.movement(which_dir)




