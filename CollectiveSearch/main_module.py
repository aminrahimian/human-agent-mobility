#!/usr/bin/env python
# coding: utf-8

# Import dependencies

# In[21]:


from __future__ import division

import random
import math
import os
# import time
# import random
import shutil
import numpy as np
from scipy import spatial as st 
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
from periodic_kdtree import PeriodicCKDTree
from sklearn.cluster import DBSCAN
from scipy import stats as sts
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pylab as pl
from IPython import display
import sys
import copy
import itertools
import matplotlib.animation as animation
import h5py

# Set global parameters

# In[22]:



digits = 3 ##the resolution for grid

radius_interaction = [1] ##the radius within which agents can detect others
epsilon = 0.001  ###
bounds = np.array([1, 1])  ## x,y length of the grid  
minstep = 0.001


# Grid class which represents the environment on which simulation takes place. It also generates targets and agents. 

# In[ ]:





# In[29]:


class Grid:


    def __init__(self, radiuss, tc, agents, beta, l, a, rho, Rv):

        # This is what builds the grid. Called every time a new grid is constructed
        # initialize the variables
        self.num_agents = agents  ## number of agents 
        self.num_targets = tc  ##number of targets
        self.beta = beta ###target distribution given Beta
        self.rho = rho # step function as social learning equation, rho is where the probability drops from 1 to 0
        self.Rv = Rv # parameyter R in varied mu search
        
        ##create targets 
        self.crea_targets()
        ###create agents with properties mu, and 
        self.crea_agents(l,a,rho,Rv)
        
        ##the radius of interaction for agents --> set to 1 
        self.rad = radiuss        
        
        ##initialize a parameter tgat checks how many targets have been consumed 
        self.deleted_targets = 0 


        #----------------------------------------------------------------------------------------------------------

    ###UPDATE:- runs agents at every time step. After agents update their position at every time step, CLUSTERING calculates the grouping between agents.
            
    def update(self):

        z=0
        r = self.rad
        while z < len(agents_list):
            agents_list[z].update(agents_list, r)
            
            z = z + 1
            
        ####run clustering 
        self.clustering(agents_list)

        
        #----------------------------------------------------------------------------------------------------------  

    ##CREATE TARGETS using power-law distribution       
    
    
    def crea_targets(self):
        self.size = 1
        # This method is in charge of creating the targets according to a resource distribution
        global targets_list
        targets_list =[]
        all_x = []
        all_y = []
        ##initialize seed population 
        for i in range(1,21) :  
            x = np.round(random.uniform(0,1)% self.size,digits)
            y = np.round(random.uniform(0,1)% self.size,digits)
            all_x.append(x) 
            all_y.append(y) 
            num = i 

            targets_list.append(Patch(x,y, num))
#         num= 0 
        self.make_targets(all_x, all_y, num)
        coord_targets = np.column_stack((all_x, all_y))
        ##create a KD Tree which contains all the targets 
        Grid.kdtree = PeriodicCKDTree(bounds, coord_targets)
        Grid.deleted_targets = 0 
        Grid.cluster_centroid = np.empty((0,2), float)
        Grid.cluster_time = 0 

    def spawntargets(self) :
        ##pick an already existed target from the list and use it as a seed to generate another target
        i = random.choice(range(len(targets_list))) 
        seeder_x = targets_list[i].x
        seeder_y = targets_list[i].y
        theta = random.uniform(0,360)
        theta = theta * (math.pi / 180)
        dis = self.levy()

        x =  np.round((seeder_x + (dis * np.cos(theta))) , digits)
        y = np.round((seeder_y + (dis * np.sin(theta))), digits )
        
        ##adjust for periodic boundaries
        if x > 1 :
            x = abs(1 - x)
        if y > 1 :
            y = abs(1 - y)
        if x < 0 :
            x = abs(1 - abs(x))
        if y < 0  :
            y = abs(1 -  abs(y)) 

        return x, y
#         

    def levy(self):
        xf = 0.1
        xi = 0.001
        mu = -self.beta
        m = mu + 1
        x = ((((xf ** m) - (xi ** m)) * (random.uniform(0,1)) ) + (xi ** m)) ** (1/m)
            
        return x 
        
    def make_targets(self, all_x, all_y, num) :
        num = num + 1
        ##continue generating targets until all the required number of targets is not achieved. 
        while len(targets_list) < self.num_targets+1 :
            x, y = self.spawntargets() 
            all_x.append(x) 
            all_y.append(y)             
            targets_list.append(Patch(x,y, num))
            num = num + 1
            
            if len(targets_list) == self.num_targets :
                break       

            
    #----------------------------------------------------------------------------------------------------------
    
    ##The following function creates agents.  
    
    
    def crea_agents(self, l, a, rho, Rv):
        ##creates agents at random locations, with 0 energy, alpha, levy
        z = 1
        global agents_list 
        agents_list =[]
        global cluster_list 
        cluster_list = []
        while len(agents_list) < self.num_agents :
            x = random.uniform(0,1)
            y = random.uniform(0,1)
            levy = l
            alpha = a
            num = z ### agent's ubique ID 
            agents_list.append(Agent(num,x,y,levy,alpha,rho,Rv))
            z = z +1 

        #----------------------------------------------------------------------------------------------------------

    ###The following function finds if an agent is located on a target
    
    def findwhichtarget(self):
        # This method helps locate a target object based on its coordinates: x, y
        
        dis, indices = Grid.kdtree.query([self.x,self.y], k=1000, distance_upper_bound = minstep)
        ###if any target is not present --> return None
        if np.isfinite(dis).any() == False:
            return None 
        
        else:
            size = np.count_nonzero(~np.isinf(dis))
            ###if more than one targets present
            if size > 1 : 
                k = 0 
                ###check which target which has food left on it.
                while k < size:
                    i = indices[k]
                    ###if a target is found with food, deplete its food and add it to the list of deleted targets
                    #####return the target ID which was consumed 
                    if targets_list[i].food == 1 :
                        targets_list[i].food = 0 
                        Grid.deleted_targets += 1 
                        return targets_list[i].num
                    k = k + 1
                ####if none of the targets have any food left, return None
                return None
            ###if only on target is present, then check if there is food on it. 
            else:
                if targets_list[0].food == 1 :
                    targets_list[0].food = 0 
                    Grid.deleted_targets += 1
                    return targets_list[0].num
                else:
                    return None
        
        
#----------------------------------------------------------------------------------------------------------------------------##

#The following functions calculate the clustering between agents after every time-step. 
    
    def clustering(self,listt):
        epsilon = 0.1
        cutoff = epsilon
        global total_clusters
        total_clusters = []
        global cluster_size
        global avg_sep
        global cluster_list
        i = 0
    ##make copy of cluster_list
        old_cluster_list = copy.deepcopy(cluster_list)
        all_x = []
        all_y=[]
        active = []
        centers = []
        all_num = []
        
        
        ##get the xy positions of the agents 
        while i < len(listt) :
            all_x.append(listt[i].x) 
            all_y.append(listt[i].y) 
            all_num.append(listt[i].num)
            i = i +1 
        ###created a matrix of x,y corodinates of all agents 
        matrix = np.column_stack((all_x, all_y))
        
        
        
        ## calculate all pairwise distances
        sep = st.distance.pdist(matrix)
        ##adjust for periodic boundaries 
       ##if the distance is >0.5, due to periodic boundaries, subtract from 1  
        sep = np.where(sep > 0.5, abs(1 - sep), sep)
        ##get the average separation between the agents 
        avg_sep += np.mean(sep)
        
        
        ##calculate clustering
        square = self.periodic_distance(matrix)
        ### use DBSCAN to detect clusters using a precomputed matrix. 
        ####minimum of 3 agents have to be present for a cluster 
        db = DBSCAN(eps=epsilon, min_samples=3, metric='precomputed').fit(square)
        cluster_labels = db.labels_
        ###get how many clusters are present at a given time step
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
       
        
        ###if clusters are present
        if num_clusters > 0 : 
            ###add to the list of total clusters
            total_clusters.append(num_clusters)
            ###get the size of clusters/number of agents in the clusters=
            counts = np.bincount(cluster_labels[cluster_labels>=0])
        
#             ###get centroids for each group 
            for z in range(num_clusters):
                points_z = matrix[cluster_labels==z,:]
                center = points_z.mean(axis=0)
                a = center
                centers.append(a)
                switch = False
                
                ###check if the clusters detected classify as new group or are old.  
                
                
                ######check if there are previous clusters
            
                if len(old_cluster_list) > 0 :
                    ##check if any of the previous clusters are active
                    if any(x.status == True for x in old_cluster_list) :
                        ## if active
                        #### go through the old cluster list and check if the centroid of the old cluster is within range 
                        #######with the new cluster detected
                        for k in range(len(old_cluster_list)):
                            if old_cluster_list[k].status == True:
                                ##check if close to new group or not 
                                dis = st.distance.euclidean(old_cluster_list[k].centroid, a)
                                ###adjust for perioidic boundaries
                                if dis > 0.5 :
                                    dis = 1 - dis
                                ####if the distance between new cluster and old cluster is less than cutoff, then count as same cluster
                                if dis <= cutoff :
                                    ###if less than cutoff, count as same cluster 
                                    
                                    #######add a unit to the duration of the cluster
                                    cluster_list[k].time += 1 
                                    #######update the size of the cluster
                                    cluster_list[k].size = counts[z]
                                    ########keep the status of the cluster active
                                    cluster_list[k].status = True 
                                    cluster_list[k].centroid = a ##update centroid
                                    active.append(k)
                                    switch = True
                                    break 
                                else: 
                                    switch = False
                                    #continue
                       ########if no previous clusters are detected OR if they are too far OR they are not active, 
                                ############make a new cluster 
                                    
                        if switch == False:         
                        ##if distance larger than cutoff, make a new cluster 
                            self.make_new_cluster(a, counts[z])
                                    
                    else: 
                        ###if no active cluster 
                        self.make_new_cluster(a, counts[z])
                ###if list is empty --> no previous cluster, create a new cluster
                else :
                    self.make_new_cluster(a, counts[z])
                    
             ##mark other existing clusters as inactive 
            all_clusters = list(range(len(old_cluster_list)))
           # print("all",all_clusters)
           # print("active", active)
            not_active = list(set(all_clusters) - set(active))
           # print(not_active)
            for s in not_active:
                cluster_list[s].status = False
                            
        else :
            ##make all existing clusters inactive 
            for x in cluster_list:
                x.status = False  

        
        
    def make_new_cluster(self, centroid, size):
        num = len(cluster_list) + 1 
        time = 1 
        status = True 
        cluster_list.append(Cluster(num, size, centroid, time ,status))
        
    def periodic_distance(self, X):
        L = 1 
        for d in range(X.shape[1]):
            # find all 1-d distances
            pd=st.distance.pdist(X[:,d].reshape(X.shape[0],1))
            # apply boundary conditions
            pd[pd>L*0.5]-=L

            try:
                # sum
                total += pd**2
            except :
                # or define the sum if not previously defined
                total=pd**2
        # transform the condensed distance matrix...
        total=pl.sqrt(total)
        # ...into a square distance matrix
        square=st.distance.squareform(total)
#         squareform(total)
        return square


#-------------------------------------------------------------------------------------------------------------------------
    ### the following function generate output files:

    
    def output(self,i):
                
        avg_eff = np.mean([c.eff for c in agents_list])
        avg_dis = np.mean([c.tdis for c in agents_list])
        avg_en = np.mean([c.food for c in agents_list])
        
#         mu = []
#         for c in agents_list:
#             data = c.alldis
#             results = powerlaw.Fit(data, xmin = 0.001)
#             mu.append(results.power_law.alpha)
        
#         avg_mu = np.nanmean(mu)
        avg_size = np.mean([c.size for c in cluster_list])
        avg_time = np.mean([c.time for c in cluster_list])
        all_clusters = len(cluster_list)
        avg_simul_clusters = np.mean(total_clusters)
        
        stringdis =  str(avg_en) + "," + str(avg_dis) + "," + str(avg_eff) +  "," + str(avg_size) + "," + str(avg_time) + "," + str(all_clusters) + "," + str(avg_simul_clusters) + "\n"
        return stringdis


# Cluster class defines the clusters/sub-groups of agents. It's properties are a unique ID, size/number of agents, duration, status(Active or not), and centroid (the central location of the cluster)

# In[24]:


class Cluster:
    def __init__(self, num, size, centroid, time, status):
        self.num = num 
        self.centroid = centroid 
        self.time = 1
        self.status = True 
        self.size = size 


# Patch class represents all targets in the environment. 
# It's properties are location(x,y), unique ID, and food (resources present, always set to 1).

# In[25]:


class Patch:
    # This class represents a target with all functions and attributes
    
    # create a target with a position, and a name
    def __init__(self, x, y, num):
        # This method is called for creating a new patch
        self.x = x
        self.y = y
        self.num = num
        self.food = 1


# 

# In[26]:


class Agent:

    def __init__(self, num, x,y, levy, alpha, rho, Rv):
        
        
        self.num = num  #ID
        self.x = x ##x-coordinate
        self.y = y  ##y-coordinate
        self.mu = levy   ###the levy exponent for individual search
        self.alpha = alpha ####social learning component
        self.rho = rho
        self.Rv = Rv
        
        self.initial_mu = levy  # store initial mu
        self.initial_alpha = alpha  # store initial alpha
        
        self.food = 0 ### food found so far
        self.target = 0  ###target on which the agent is on 
        self.detected_target = None  # track detected target
        
        self.eff = 0 ###search efficiency
        self.numcoltar = 0 ### number of collected targets
        
        
        self.rw_switch = 0 ###conducting a random walk or not-- boolean switch
        self.tw_switch = 0 ###conducting a targeted walk towards another agent or not -- boolean switch 
               
        self.d = 0 
        self.theta = 0 
        self.tdis = 1
        
        self.tw_step_length = 0 ### variable to accumulate continuous Targeted Walk step length
        self.tw_step_lengths = []  # List to store the completed TW step lengths
        self.tw_collecting = False ### Flag to indicate if we are collecting TW step length
        
        #----------------------------------------------------------------------------------------------------------

    def update(self, agents_list,r):  ##check if on a target or not
       
        ###update search efficiency at every time-step
        if self.food == 0 :
            self.eff = 0
            self.numcoltar = 0 ### number of collected targets
        else : 
            self.eff = self.food / self.tdis * minstep
            self.numcoltar = self.food
            
        ##check if the agent is on a target
        num = Grid.findwhichtarget(self)
        
        ###if there is target on current location:
        if num != None:            
            self.target = num   ###update which target the agent is on 
            self.food = self.food + 1  ##update the total number of targets/food found
            
            self.detected_target = (self.x, self.y)  # mark detected target
            self.mu = 3
            self.alpha = 20
            
            ### terminate both walks 
            self.rw_switch = 0 
            self.tw_switch = 0 
            self.walk = 0 
            
            
        # If there is no food, then look for others 
        #if P(scrounge) = 1, if rw, terminate rw walk and start a targeted walk towards neighbor
        # if P(scrounge), if tw, terminate current walk and start moving in a new direction towards another neighbor 
        #if P(scrounge) = 0 , if rw, continue 
        ## if P(scrounge) = 0, if tw, continue
        
        else : 
            ###look for others
            self.target = 0
            if self.detected_target:
                distance_to_target = st.distance.euclidean(self.detected_target, (self.x, self.y))
                if distance_to_target > self.Rv: #parameter R in varied mu search model
                    self.mu = self.initial_mu
                    self.alpha = self.initial_alpha
                    self.detected_target = None
                    
            #####if already walking towards another agent
            if self.tw_switch == 1 :
                newtarget = self.check_forothers(agents_list,  r)
                ####if new agent is detected, start a new TW
                if newtarget != None : 
                    ###if distance to new agent is more than 0, take a step
                    if self.d > 0 :
                        self.step()
                        ####if after taking a step, agent has reached the destination, terminate the new TW
                        if self.d <= 0 :
                            self.tw_switch = 0
                    ###if the distance to another agent is 0, terminate the new TW       
                    else : 
                        self.tw_switch = 0 
                
                
                
                #######if no new agent is detected, continue previous TW
                else :   
                    if self.d > 0 :
                        self.step()
                        if self.d <= 0 :
                            self.tw_switch = 0
                    else : 
                        self.tw_switch = 0 
                            
            
            #######if the agent was not doing a TW or RW    
            elif self.tw_switch == 0 and self.rw_switch == 0 :
                ###check for other agents 
                newtarget = self.check_forothers(agents_list, r)
                ####if an agent is detected, start a TW 
                if newtarget != None :   
                    self.tw_switch = 1
                    self.rw_switch = 0 
                    if self.d > 0 :
                        self.step()
                        if self.d <= 0 :
                            self.tw_switch = 0
                    else : 
                        self.tw_switch = 0 
                        
                ####if no new agent is detected, start a random walk (RW)
                else : 
                    rw_update = self.rw()  ##start a rw
                    self.d = rw_update[0]  ###get the distance to move
                    self.theta = rw_update[1]  ### get the direction 
                    if self.d > 0 :
                        self.step()
                        self.rw_switch = 1
                        if self.d <= 0 :
                            self.rw_switch = 0
                    else : 
                        self.rw_switch = 0 
                            
                    
            #####if the agent was already doing a random walk 
            elif self.rw_switch == 1 : 
                ###check for other agents 
                newtarget = self.check_forothers(agents_list, r)
                ###if a new agent deteced, terminate RW and start TW 
                if newtarget != None :   ##stop rw and do tw instead
                    self.tw_switch = 1
                    self.rw_switch = 0 
                    if self.d > 0 :
                        self.step()
                        if self.d <= 0 :
                            self.tw_switch = 0
                    else : 
                        self.tw_switch = 0 
                
                ####Else if no agent detected, continue the random walk 
                else : 
                    if self.d > 0 :   ##continue rw
                        self.step()
                        if self.d <= 0 :
                            self.rw_switch = 0
                    else : 
                        self.rw_switch = 0
            
            # If the agent is currently doing a targeted walk, continue to accumulate
            if self.tw_switch == 1:
                # Continue TW, already handled in the `step()` method
                pass
            elif self.tw_collecting:
                # The targeted walk has just ended, stop collecting
                self.tw_collecting = False
                self.tw_step_lengths.append(self.tw_step_length)  # Store the TW step length
                self.tw_step_length = 0  # Reset after storing
                                  
            
    

            
            
    ####this function calculates the distance for a random walk based on Levy distribution 
    def calc_dist(self):
        xf = 1
        if self.mu == 'random' :
            x = minstep
            return x 
        elif self.mu == 'straight':
            x = 1000000000 
            return x 
    
        else:
            xi = minstep
            mu = -self.mu
            m = mu + 1
            x = ((((xf ** m) - (xi ** m)) * (random.uniform(0,1)) ) + (xi ** m)) ** (1/m)
            if self.mu == 'random' :
                x = minstep
            return x
 
 
      #----------------------------------------------------------------------------------------------------------
     
    def rw(self) :     
        r = self.calc_dist() 
        theta = random.uniform(0,360)
        return( r, theta)

    ###this function updates agent's position,one step at a time 
    def step(self) : 
        
        if self.d <= minstep :
            r = self.d
        else :
            r = minstep
        theta = self.theta * (math.pi / 180)
        x2 = (self.x + (r * np.cos(theta)))
        y2 = (self.y + (r * np.sin(theta)))
        ###adjust for periodic boundaries
        if x2 >= 1 :
            x2 = abs(1 - x2)
        if y2 >= 1 :
            y2 = abs(1 - y2)
        if x2 <= 0 :
            x2 = abs(1 - abs(x2))
        if y2 <= 0  :
            y2 = abs(1 -  abs(y2))     
        self.x = x2
        self.y = y2
        ###update how much distance is left to move
        self.d = self.d - r
        #####update the total distance traveled by an agent 
        self.tdis = self.tdis + r
        
        # Accumulate the step length for a continuous targeted walk
        if self.tw_switch == 1:
            if not self.tw_collecting:
                self.tw_collecting = True # start collecting for this TW
                self.tw_step_length = 0 # reset the length
            self.tw_step_length += r # add the current step length to the total TW length
        else:
            if self.tw_collecting:
                self.tw_collecting = False  # Stop collecting, walk is finished
                self.tw_step_lengths.append(self.tw_step_length)  # Store the TW step length
                self.tw_step_length = 0  # Reset after storing
        
    #### this function checks for agents that are on a target and generates a TW based on alpha 
    
    def check_forothers(self, agents_list,  r):
        z = 0 
        all_x = []
        all_y = []
        creepable = []
        coord_others = []
        indices =[]
        i = 0 
               
         ###get other agents who are on a target and not yourself 
        
        for i in range(len(agents_list)):  
            if agents_list[i].target != 0 :
                creepable.append(i)
                     
                    
                
        ###if there are agents that are on a target 
        if len(creepable) > 0 : 
            z = 0 
            for z in creepable :
                all_x.append(agents_list[z].x) 
                all_y.append(agents_list[z].y) 
                
            ###create a matrix for their coordinates
            coord_others = np.column_stack((all_x, all_y))
            
            ###find which agent is the closest
            dis, indices = self.distance(coord_others, (self.x, self.y), r)
            ###if no agent is present or outside radius, return None 
            if math.isinf(dis) == True :
                return None 
            ####else 
            else :
                dis2 = (dis *1000)
                chance = random.uniform(0,1)         
                ni= creepable[indices]   
                ####if already doing a TW 
                
                if self.tw_switch == 1 :
                ###check if current distance to previous agent is more than the distance to new agent 
                    if self.d > dis:
                        ##get the coordinates of the new agent 
                        new_coord = (agents_list[ni].x, agents_list[ni].y)
                        self.d = dis ###update distance for walk 
                        self.theta = self.tw_angle(self.x, self.y, new_coord[0], new_coord[1]) ###update angle for walk 
                        to_return = new_coord[0],new_coord[1], dis
                    else : 
                        return None 
                ###if currently not doing a TW 
                else:
                    # if chance < math.exp(- self.alpha * dis2**2*5): # step function
                    if dis < self.rho: # step function
                        new_coord = (agents_list[ni].x, agents_list[ni].y)
                        self.d = dis 
                        self.theta = self.tw_angle(self.x, self.y, new_coord[0], new_coord[1])
                        to_return = new_coord[0],new_coord[1], dis
                        return to_return
                    else : 
                        return None
            
        else : 
            return None
   


    def tw_angle(self, x1, y1, x2, y2):
        ###calculate angle between two agents 
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx > 0.5 :
            if x1 > 0.5 :
                x2 = 1 + (x2 - 0)
            else :
                x2 = 0 - (1 - x2)
        if dy > 0.5 :
            if y1 > 0.5 :
                y2 = 1 + (y2 - 0)
            else :
                y2 = 0 - (1 - y2) 
        dx = x2 - x1
        dy = y2 - y1
        theta = math.degrees(math.atan2(dy, dx))
        if (theta < 0.0) :
            theta += 360.0
        return theta
   
        
       
        
    def distance(self, x0, x1, upperbound):
        ####this function checks if the agents deteccted are within radius 
        delta = np.abs(x0 - x1)
        #####adjust for periodic boundaries
        delta = np.where(delta > 0.5 * bounds, delta - bounds, delta)
        alld =  np.sqrt((delta ** 2).sum(axis=-1))
        alld = list(map(lambda x: np.inf if x > upperbound else x, alld))
        ####return the distance to the closest agent and the ID of the closest agent 
        return np.min(alld), np.argmin(alld)

        
            
        
    


# In[27]:


def main_novideo(params):    
#     from Grid import Grid
    radiuss = params[3]
    tc = params[0]
    mono = params[1]
    beta = params[2]
    l = params[4]
    a = params[5]
    rho = params[6]
    Rv = params[7]
    
    global gen_counter 
    global food_found
    global avg_sep 
    avg_sep = 0 

    gen_counter = 0 
    food_found = 0 
    
    
    grid = Grid(radiuss, tc, mono, beta, l, a, rho, Rv)
    i = 1
    # Main cycle of each iteration
    all_rep = ""
    mainn = ""
    alphstr = ""
    levystr = ""
    targetslist=""
     
    string2=""
     
    
    pf = 0 
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3= []
    y3 = []

    
    while pf < 30 : 
        pf = ( Grid.deleted_targets / (tc * 0.01))
        grid.update()
        i = i + 1
        
    ###when the simulation is done, get the result 
    string = grid.output(i)
    
    string2 = str(i) + "," +str(tc)+","+str(mono)+","+str(beta)+","+str(radiuss)+","+str(l) + "," + str(rho) + "," + string
    return string2

        
        
def main_video(params,fig, ax, tick_speed, video_filename):    
#     from Grid import Grid
    radiuss = params[3]
    tc = params[0]
    mono = params[1]
    beta = params[2]
    l = params[4]
    a = params[5]
    rho = params[6]
    Rv = params[7]

    global gen_counter 
    global food_found
    global avg_sep 
    avg_sep = 0 

    gen_counter = 0 
    food_found = 0 
    
    
    grid = Grid(radiuss, tc, mono, beta, l, a, rho, Rv)
    i = 1
    # Main cycle of each iteration
    all_rep = ""
    mainn = ""
    alphstr = ""
    levystr = ""
    targetslist=""
     
    string2=""
     
    
    pf = 0 
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3= []
    y3 = []
    # Set up video writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    with writer.saving(fig, video_filename, 100):
        while pf < 30 : 
            pf = ( Grid.deleted_targets / (tc * 0.01))
            grid.update()
            if (i % tick_speed) == 0 : 
                ax.clear()
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                tr_x = []
                tr_y = []
    
                tr_color = []
                for k in targets_list:
                    if k.food == 1 :
                        tr_x.append(k.x)
                        tr_y.append(k.y)
                ag_x = []
                ag_y = []
                ag_al=[]
                ag_size=[]
                ag_color = []
                ag_rw = []
                ag_tw = []
                ag_tx = []
                ag_ty = []
                ag_mu = []
                ag_eff = []
                edge_color = []
                ag_theta = []
                for k in agents_list:
                    ag_x.append(k.x)
                    ag_y.append(k.y)
                    ag_al.append(k.alpha)
                    ag_rw.append(k.rw_switch)
                    ag_tw.append(k.tw_switch)
                    ag_eff.append(k.eff)
    
                    theta = (90 - k.theta) % 360
    #                 theta = k.theta
                    ag_theta.append(theta)
    #                 print(k, theta)
                    if k.target != 0:
                        ag_color.append('red')   ###personal info
                    elif k.rw_switch == 1 :
                        ag_color.append('blue')
                    elif k.tw_switch == 1:
                        ag_color.append('purple')
                    else :
                        ag_color.append('black')
    
    
                ####scatter patches in green 
                ax.scatter(tr_x, tr_y, c = 'green', s=5 )
    
    
                g = 0 
                ###Scatter agents 
                for g in range(len(ag_x)):
                    ax.scatter(ag_x[g], ag_y[g],  marker=(3, 0, ag_theta[g]), c=ag_color[g], s = 100)
    #                 ax.scatter(ag_x[g], ag_y[g],  marker=(3, 0, ag_theta[g]), c='blue', s = 40)
                plt.pause(0.01)
                plt.title('percent_found:'+str(pf)+';tick:'+str(i)+';avg_efficiency:'+str(np.format_float_positional(np.mean(ag_eff), precision=3)) )
                fig.canvas.draw()
                
                # Save the current frame
                writer.grab_frame()
                
                fig.show()
                ax.cla()
            i = i + 1
        
    ###when the simulation is done, get the result 
    string = grid.output(i)
    
    string2 = str(i) + "," +str(tc)+","+str(mono)+","+str(beta)+","+str(radiuss)+","+str(l) + "," + str(rho) + "," + string
    return string2

def main_timeana(params, tick_speed, hdf5_filename, case_id):
    radiuss = params[3]
    tc = params[0]
    mono = params[1]
    beta = params[2]
    l = params[4]
    a = params[5]
    rho = params[6]
    Rv = params[7]

    global gen_counter 
    global food_found
    global avg_sep 
    avg_sep = 0 

    gen_counter = 0 
    food_found = 0 
    
    
    grid = Grid(radiuss, tc, mono, beta, l, a, rho, Rv)
    i = 1
    # Main cycle of each iteration
    all_rep = ""
    mainn = ""
    alphstr = ""
    levystr = ""
    targetslist=""
     
    string2=""
     
    # Lists to store ticks and average efficiencies per timestep
    ticks = []
    avg_foods = []
    avg_efficiencies = []
    all_mu = []
    all_tw_step_lengths = []  # New list to store step lengths for all continuous TWs

    pf = 0 
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3= []
    y3 = []

    
    while pf < 30 : 
        pf = ( Grid.deleted_targets / (tc * 0.01))
        grid.update()
        if (i % tick_speed) == 0 : 
            ag_eff = []
            ag_tar = []
            ag_mu = []
            tw_step = []
            for k in agents_list:
                ag_eff.append(k.eff)
                ag_tar.append(k.numcoltar)
                ag_mu.append(k.mu)
                # Collect all continuous TW step lengths
                tw_step.append(k.tw_step_length)
                    
            avg_efficiency = np.mean(ag_eff)
            avg_food = np.sum(ag_tar)
            ticks.append(i)  # Save the current tick
            avg_efficiencies.append(avg_efficiency)  # Save the average efficiency
            avg_foods.append(avg_food)
            all_mu.append(ag_mu)
            all_tw_step_lengths.append(tw_step)
        i = i + 1
    
    # Save the data to the HDF5 file
    with h5py.File(hdf5_filename, 'a') as hdf5_file:
        # List all groups
        print("Keys: %s" % hdf5_file.keys())
        # Create a group for each case using the case_id
        group_name = f"case_{case_id}"
        case_group = hdf5_file.create_group(group_name)
        # Create a subgroup within "group1" called "subgroup1"
        mu_group = case_group.create_group("subgroup_mu")
        mu_group.create_dataset('mu', data=np.array(all_mu), compression="gzip", compression_opts=9)
        
        tw_group = case_group.create_group('tw_step_lengths')
        tw_group.create_dataset('step', data=np.array(all_tw_step_lengths), compression="gzip", compression_opts=9)
        
        # Save ticks and efficiencies as datasets
        case_group.create_dataset('ticks', data=np.array(ticks), compression="gzip", compression_opts=9)
        case_group.create_dataset('foods', data=np.array(avg_foods), compression="gzip", compression_opts=9)
        case_group.create_dataset('average_efficiencies', data=np.array(avg_efficiencies), compression="gzip", compression_opts=9)
        case_group.create_dataset('TW_step_lengths', data=np.array(all_tw_step_lengths), compression="gzip", compression_opts=9)
        
        # # Save tw_step_lengths as strings
        # tw_group = case_group.create_group('tw_step_lengths')
        # dt = h5py.vlen_dtype(np.dtype('float64'))
        # tw_group.create_dataset('tw_step', data=np.array(all_tw_step_lengths, dtype=dt), compression="gzip", compression_opts=9)

        # tw_group.create_dataset('tw_step', data=np.array(all_tw_step_lengths), compression="gzip", compression_opts=9)
        
        # You can also save the parameters for reference
        case_group.attrs['parameters'] = str(params)
    
    ###when the simulation is done, get the result 
    string = grid.output(i)
    
    string2 = str(i) + "," +str(tc)+","+str(mono)+","+str(beta)+","+str(radiuss)+","+str(l) + "," + str(rho) + "," + string
    return string2

def main_video_timedata(params,fig, ax, tick_speed, video_filename, hdf5_filename, case_id):    
#     from Grid import Grid
    radiuss = params[3]
    tc = params[0]
    mono = params[1]
    beta = params[2]
    l = params[4]
    a = params[5]
    rho = params[6]
    Rv = params[7]

    global gen_counter 
    global food_found
    global avg_sep 
    avg_sep = 0 

    gen_counter = 0 
    food_found = 0 
    
    
    grid = Grid(radiuss, tc, mono, beta, l, a, rho, Rv)
    i = 1
    # Main cycle of each iteration
    all_rep = ""
    mainn = ""
    alphstr = ""
    levystr = ""
    targetslist=""
     
    string2=""
     
    # Lists to store ticks and average efficiencies per timestep
    ticks = []
    avg_foods = []
    avg_efficiencies = []
    
    pf = 0 
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3= []
    y3 = []
    # Set up video writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    with writer.saving(fig, video_filename, 100):
        while pf < 30 : 
            pf = ( Grid.deleted_targets / (tc * 0.01))
            grid.update()
            if (i % tick_speed) == 0 : 
                ax.clear()
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                tr_x = []
                tr_y = []
    
                tr_color = []
                ag_eff = []
                ag_tar = []
                for k in targets_list:
                    if k.food == 1 :
                        tr_x.append(k.x)
                        tr_y.append(k.y)
                ag_x = []
                ag_y = []
                ag_al=[]
                ag_size=[]
                ag_color = []
                ag_rw = []
                ag_tw = []
                ag_tx = []
                ag_ty = []
                ag_mu = []
                ag_eff = []
                edge_color = []
                ag_theta = []
                for k in agents_list:
                    ag_x.append(k.x)
                    ag_y.append(k.y)
                    ag_al.append(k.alpha)
                    ag_rw.append(k.rw_switch)
                    ag_tw.append(k.tw_switch)
                    ag_eff.append(k.eff)
                    ag_tar.append(k.numcoltar)
    
                    theta = (90 - k.theta) % 360
    #                 theta = k.theta
                    ag_theta.append(theta)
    #                 print(k, theta)
                    if k.target != 0:
                        ag_color.append('red')   ###personal info
                    elif k.rw_switch == 1 :
                        ag_color.append('blue')
                    elif k.tw_switch == 1:
                        ag_color.append('purple')
                    else :
                        ag_color.append('black')
    
    
                ####scatter patches in green 
                ax.scatter(tr_x, tr_y, c = 'green', s=5 )
    
    
                g = 0 
                ###Scatter agents 
                for g in range(len(ag_x)):
                    ax.scatter(ag_x[g], ag_y[g],  marker=(3, 0, ag_theta[g]), c=ag_color[g], s = 100)
    #                 ax.scatter(ag_x[g], ag_y[g],  marker=(3, 0, ag_theta[g]), c='blue', s = 40)
                plt.pause(0.01)
                plt.title('percent_found:'+str(pf)+';tick:'+str(i)+';avg_efficiency:'+str(np.format_float_positional(np.mean(ag_eff), precision=3)) )
                fig.canvas.draw()
                
                # Save the current frame
                writer.grab_frame()
                
                fig.show()
                ax.cla()
                
                avg_efficiency = np.mean(ag_eff)
                avg_food = np.sum(ag_tar)
                ticks.append(i)  # Save the current tick
                avg_efficiencies.append(avg_efficiency)  # Save the average efficiency
                avg_foods.append(avg_food)
            i = i + 1
    
    # Save the data to the HDF5 file
    with h5py.File(hdf5_filename, 'a') as hdf5_file:
        # Create a group for each case using the case_id
        group_name = f"case_{case_id}"
        case_group = hdf5_file.create_group(group_name)
        # Save ticks and efficiencies as datasets
        case_group.create_dataset('ticks', data=np.array(ticks), compression="gzip", compression_opts=9)
        case_group.create_dataset('foods', data=np.array(avg_foods), compression="gzip", compression_opts=9)
        case_group.create_dataset('average_efficiencies', data=np.array(avg_efficiencies), compression="gzip", compression_opts=9)
        # You can also save the parameters for reference
        case_group.attrs['parameters'] = str(params)
    
    ###when the simulation is done, get the result 
    string = grid.output(i)
    
    string2 = str(i) + "," +str(tc)+","+str(mono)+","+str(beta)+","+str(radiuss)+","+str(l) + "," + str(rho) + "," + string
    return string2


def target_structure(params):
#     from Grid import Grid
    radiuss = params[3]
    tc = params[0]
    mono = params[1]
    beta = params[2]
    l = params[4]
    a = params[5]
    rho = params[6]
    Rv = params[7]
    
    grid = Grid(radiuss, tc, mono, beta, l, a, rho, Rv)
    grid.crea_targets()
    # Access the generated targets
    tar_x = [];
    tar_y = [];
    for target in targets_list:
        # print(f"Target ID: {target.num}, X: {target.x}, Y: {target.y}")
        tar_x.append(target.x)
        tar_y.append(target.y)
    return tar_x,tar_y 

