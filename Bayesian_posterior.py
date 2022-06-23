from scipy.stats import norm
import numpy as np
import copy
from scipy.spatial import distance
from scipy.integrate import quad
import scipy.optimize as optimize
from scipy.optimize import minimize
from scipy.integrate import dblquad
from scipy import integrate
from numpy import linalg as LA
import pandas as pd
import random
from random import sample
# import matplotlib.pyplot as plt



# prior_theta1=5
# prior_theta2=5
# sd_prior=2

# set_observations=np.array([[1,2,0],[2,1,0], [4,3,0],[7,6,1],[9,8,1]])




def evaluation_theta(set_observations,theta1, theta2, prior_theta1, prior_theta2,sd_prior):
    
    
    f_theta1=norm.pdf(theta1, loc=prior_theta1, scale=sd_prior)
    f_theta2=norm.pdf(theta1, loc=prior_theta2, scale=sd_prior)
    
    temp=1
    
    for i in range(int(set_observations.shape[0])):
        
        if set_observations[-1,2]==1:
            
            temp*= np.exp(-(distance.euclidean((set_observations[i,0],set_observations[i,1] ), (theta1,theta2)))*(1/50)) 
            
        else:
    
            temp*= 1-np.exp(-(distance.euclidean((set_observations[i,0],set_observations[i,1] ), (theta1,theta2)))*(1/50))
    
    
    
    return temp*f_theta1*f_theta2




def circle_elimination(theta1,theta2,set_observations, Radious):
    
    temp_bol=True
    
    for i in range(int(set_observations.shape[0])):
        
        
        if distance.euclidean((set_observations[i,0],set_observations[i,1]),(theta1,theta2))<=50:
            
            temp_bol=temp_bol&False
    
        else:
            
            temp_bol=temp_bol&True
            
    
    return temp_bol


def choosing_direction(nn,L,H):
    
    Radious=50
    prior_theta1=L*(0.5)
    prior_theta2=H*(0.5)
    sd_prior=800
    

    
    file_posterior='input'+ '_'+str(int(nn))+'.csv'
    data_historical=pd.read_csv(file_posterior, header=None)  
    set_observations=data_historical.to_numpy()
    
    ones_array=set_observations[set_observations[:,2]==1]
    zeros_array=set_observations[set_observations[:,2]==0]
    
    
    if ((set_observations.shape[0] -ones_array.shape[0])>75):
       
        
       hj=zeros_array.shape[0]
       
       new_zeros_array=zeros_array[sample(list(range(hj)),75),:]
        
       sub_set_observations=np.concatenate((new_zeros_array, ones_array), axis=0)   
      
        
        
    else:
        
        sub_set_observations=set_observations

    
    x1=set_observations[-1,0]
    x2=set_observations[-1,1]
   
    
    
    if ((x1<0)&((x2>=0) & (x2<=L))):
        
        print("Wall")
        
        return sample([1,2,8],1)[0]
    
    elif ((x1<0)&(x2<0)):
        
        print("Wall")
        
        return 2
    
    elif (((x1>=0) & (x1<=L))&(x2<0)):
        
        print("Wall")
        
        return sample([2,3,4],1)[0]
    
    elif ((x1>=L)&(x2<0)):
        
        print("Wall")
        
        return 4
    
    elif ((x1>=L)&((x2>=0) & (x2<=L))):
        
        print("Wall")
        
        return sample([4,5,6],1)[0]
    
    elif ((x1>=L)&(x2>=H)):
        
        
        print("Wall")
        
        return 6
    
    elif (((x1>=0) & (x1<=L))&(x2>=H)):
        
        print("Wall")
        
        return sample([6,7,8],1)[0]
    
    elif ((x1<0)&(x2>=L)):
        
        print("Wall")
        
        return 8
    
    
    lgth=166

    
    comparison_table=np.array([[x1 +lgth,x2,0 ],
                                [x1 +lgth*np.cos(np.pi*0.25), x2+lgth*np.sin(np.pi*0.25),0],
                                [x1, x2 +lgth, 0],
                                [x1-lgth*np.cos(np.pi*0.25), x2+lgth*np.cos(np.pi*0.25),0],
                                [x1-lgth, x2,0],
                                [x1-lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0],
                                [x1, x2-lgth, 0],
                                [x1+lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0]])
    
    
    
    
    def posterior_evaluation(theta1,theta2):
        
    
        
        # =Theta_vector[0]
        # theta2=Theta_vector[1]
        
        f_theta1=norm.pdf(theta1, loc=prior_theta1, scale=sd_prior)
        f_theta2=norm.pdf(theta1, loc=prior_theta2, scale=sd_prior)
        
        temp=1
        
        for i in range(int(sub_set_observations.shape[0])):
            
            if sub_set_observations[i,2]==1:
                
                temp*= np.exp(-(distance.euclidean((sub_set_observations[i,0],sub_set_observations[i,1] ), (theta1,theta2)))*(1/500)) 
                
            else:
        
                temp*= 1-np.exp(-(distance.euclidean((sub_set_observations[i,0],sub_set_observations[i,1] ), (theta1,theta2)))*(1/500))
        
        
        
        
        return temp*f_theta1*f_theta2

    
    # x_min = optimize.minimize(posterior_evaluation, x0=[L*0.75, H*0.75])
    
    # x0=[750,750]
    
    
    # bnds = ((0, 2000), (0, 2000))
    # res = minimize(posterior_evaluation, x0, bounds=bnds ,method='Nelder-Mead', tol=1e-6) 
    
    
    montecarlo=np.zeros((500,3))
    montecarlo[:,0]=np.random.uniform(0.0, 1.0, size = (500,))*2000
    montecarlo[:,1]=np.random.uniform(0.0, 1.0, size = (500,))*2000
    
    
    for k in range(500):
        
        montecarlo[k,2]=posterior_evaluation(montecarlo[k,0],montecarlo[k,1])
    
    
    index_closer=np.argmax(montecarlo[:,2])
    
    
    for j in range(8):
        
        theta1 = comparison_table[j,0]
        theta2 = comparison_table[j,1]
        
        comparison_table[j,2] = distance.euclidean((comparison_table[j,0],comparison_table[j,1]),(montecarlo[index_closer,0],montecarlo[index_closer,1]))
        
        if not circle_elimination(theta1,theta2,set_observations, Radious):
            
            comparison_table[j,2]=1e10
        

    min_val = min(comparison_table[:,2])
    

    
    if min_val==1e10:
               
                 
        for j in range(8):
            
            theta1 = comparison_table[j,0]
            theta2 = comparison_table[j,1]
            
            comparison_table[j,2] = distance.euclidean((comparison_table[j,0],comparison_table[j,1]),(montecarlo[index_closer,0],montecarlo[index_closer,1]))
            
            # if not circle_elimination(theta1,theta2,set_observations, Radious):
                
            #     comparison_table[j,2]=1e10
            
    
        min_val = min(comparison_table[:,2])
        
        return (np.argmin(comparison_table[:,2])+1)
        
        
        # print("random")
        
        # if ((x1>=1000)&(x2>=1000)):
            
        #     return 5
        
        # elif ((x1<=1000)&(x2>=1000)):
        #     return 7
        
        # elif ((x1<=1000)&(x2<=1000)):
        #     return 1
        
        # elif ((x1>=1000)&(x2<=1000)):
            
        #     return 3
            
      
    else:
        
        
        print("ongoing")
        return np.argmin(comparison_table[:,2])+1

        
        
            

theta=choosing_direction(nn,L,H) 
theta = float(theta)




# plt.scatter(set_observations[:,0], set_observations[:,1])
# plt.show()

# plt.scatter(montecarlo[:,0], montecarlo[:,1])
# plt.show()
