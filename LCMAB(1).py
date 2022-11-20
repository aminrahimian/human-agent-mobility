import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
# import networkx as nx
# pip install networkx
import pickle
import itertools
from numpy.random import choice




epsilon=100

def evaluation_theta(set_observations,theta1, theta2, prior_theta1, prior_theta2,sd_prior):
    
    
    f_theta1=norm.pdf(theta1, loc=prior_theta1, scale=sd_prior)
    f_theta2=norm.pdf(theta2, loc=prior_theta2, scale=sd_prior)
    
    temp=0
    
    for i in range(int(set_observations.shape[0])):
        
        if set_observations[i,2]==1:
            
            temp+= (-(distance.euclidean((set_observations[i,0],set_observations[i,1] ), (theta1,theta2)))*(1/50)) 
            
        else:
    
            temp+= np.log(1-np.exp(-(distance.euclidean((set_observations[i,0],set_observations[i,1] ), (theta1,theta2)))*(1/50)))
    
    
    
    return temp+ np.log(f_theta1) +np.log(f_theta2)

# L=3000
# H=3000
# t1=2
# t2=2

# nn=264


def my_indices(lst, item):
   return [i for i, x in enumerate(lst) if x == item]





def already_covered(coordinates, set_observations):
    
    was_in=False
    
    
    for i in range(int(set_observations.shape[0])):

        if distance.euclidean((set_observations[i,0],set_observations[i,1]),(coordinates[0],coordinates[1])) <= epsilon:
            
           was_in=True
            
           break
        
        
    return was_in



def already_in_the_local(d1,d2,local_opt):
    
    answ=False
    
    if distance.euclidean((d1,d2),(local_opt[0],local_opt[1])) <= epsilon:
        
        answ=True
        
    
    return answ
    
    
    

def choosing_direction(nn,L,H,t1,t2):
    
    
    if nn==1:
        
    
        
    
        file_posterior='input'+ '_'+str(int(nn))+'.csv'
        data_historical=pd.read_csv(file_posterior, header=None)  
        set_observations=data_historical.to_numpy()
        
        x1=set_observations[-1,0]
        x2=set_observations[-1,1]
        
        lgth=168
        
        
        comparison_table=np.array([[x1 +lgth,x2,0 ],
                                    [x1 +lgth*np.cos(np.pi*0.25), x2+lgth*np.sin(np.pi*0.25),0],
                                    [x1, x2 +lgth, 0],
                                    [x1-lgth*np.cos(np.pi*0.25), x2+lgth*np.cos(np.pi*0.25),0],
                                    [x1-lgth, x2,0],
                                    [x1-lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0],
                                    [x1, x2-lgth, 0],
                                    [x1+lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0]])
        
        
        
        linespacing=int(np.ceil((L+L*0.05)*np.sqrt(2)/epsilon))
        
    
        
        # mc_theta1=np.random.uniform(0.0, 1.0, size = (linespacing,))*L*1.12
        # mc_theta2=np.random.uniform(0.0, 1.0, size = (linespacing,))*L*1.12
        
        
        
        mc_theta1=np.linspace(10, L,linespacing , endpoint=True)
        mc_theta2=np.linspace(10, L,linespacing , endpoint=True)
        
        all_coordinates=list(itertools.product(mc_theta1, mc_theta2))
    
        eval_coordinates=pd.DataFrame(np.zeros((len(all_coordinates),2)))
        eval_coordinates[0]=all_coordinates
    
        prior_theta1=L*0.5
        prior_theta2=L*0.5
        sd_prior=L
        
        col2=[evaluation_theta(set_observations,eval_coordinates.iat[i,0][0],eval_coordinates.iat[i,0][1], prior_theta1, prior_theta2,sd_prior) for i in range(eval_coordinates.shape[0])]
        eval_coordinates[1]=col2
        
        
        list_of_candidates=np.array(eval_coordinates[0])
        
        probability_distribution=np.array(eval_coordinates[1])/(np.sum(eval_coordinates[1]))
        
        draw = choice(list_of_candidates, 1, p=probability_distribution)[0]
        
    
        
    
        
        with open('eval_coordinates.pkl', 'wb') as f:
           
           pickle.dump(eval_coordinates, f)
       
       
        local_opt=[draw[0],draw[1] ]
             
        with open('local_opt.pkl', 'wb') as f:
            
            pickle.dump(local_opt, f)
        
        
        for i in range(8):
            
            comparison_table[i,2]=distance.euclidean((comparison_table[i,0],comparison_table[i,1]),(local_opt[0], local_opt[1]))
        
        
        which_dir=np.argmin(comparison_table[:,2])
        
            
    
    else:
        
      
        
        with open('local_opt.pkl', 'rb') as f:
            
            local_opt = pickle.load(f)
        
        
        file_posterior='input'+ '_'+str(int(nn))+'.csv'
        data_historical=pd.read_csv(file_posterior, header=None)  
        set_observations=data_historical.to_numpy()
        
        x1=set_observations[-1,0]
        x2=set_observations[-1,1]
        
        lgth=168
        
        
        comparison_table=np.array([[x1 +lgth,x2,0 ],
                                    [x1 +lgth*np.cos(np.pi*0.25), x2+lgth*np.sin(np.pi*0.25),0],
                                    [x1, x2 +lgth, 0],
                                    [x1-lgth*np.cos(np.pi*0.25), x2+lgth*np.cos(np.pi*0.25),0],
                                    [x1-lgth, x2,0],
                                    [x1-lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0],
                                    [x1, x2-lgth, 0],
                                    [x1+lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0]])
        
        
        
        
        prior_theta1=L*0.5
        prior_theta2=L*0.5
        sd_prior=L
        
        
        if not already_in_the_local(x1, x2, local_opt):
            
              
            for i in range(8):
                
                comparison_table[i,2]=distance.euclidean((comparison_table[i,0],comparison_table[i,1]),(local_opt[0], local_opt[1]))
            
            
            which_dir=np.argmin(comparison_table[:,2])
        
    
        
        else:
            
            with open('eval_coordinates.pkl', 'rb') as f:
                
                eval_coordinates = pickle.load(f)
                
            
            boolean_mask=[not already_covered(eval_coordinates.iat[i,0], set_observations) for i in range(eval_coordinates.shape[0])]
            eval_coordinates=eval_coordinates[boolean_mask]
            new_index=list(range(eval_coordinates.shape[0]))
            eval_coordinates.index= new_index
            
            
            col2=[evaluation_theta(set_observations,eval_coordinates.iat[i,0][0],eval_coordinates.iat[i,0][1], prior_theta1, prior_theta2,sd_prior) for i in range(eval_coordinates.shape[0])]
            eval_coordinates[1]=col2
            
            
            list_of_candidates=np.array(eval_coordinates[0])
            
            probability_distribution=np.array(eval_coordinates[1])/(np.sum(eval_coordinates[1]))
            
            draw = choice(list_of_candidates, 1, p=probability_distribution)[0]
            
        
            
            with open('eval_coordinates.pkl', 'wb') as f:
               
               pickle.dump(eval_coordinates, f)
               
               
            
            local_opt=[draw[0],draw[1] ]
                 
            with open('local_opt.pkl', 'wb') as f:
                
                pickle.dump(local_opt, f)
            
            
            for i in range(8):
                
                comparison_table[i,2]=distance.euclidean((comparison_table[i,0],comparison_table[i,1]),(local_opt[0], local_opt[1]))
            
            
            which_dir=np.argmin(comparison_table[:,2])
            
            
        
    
    return (which_dir+1, 2)
        

theta = choosing_direction(nn,L,H,t1,t2)
theta = (float(theta[0]),theta[1])






