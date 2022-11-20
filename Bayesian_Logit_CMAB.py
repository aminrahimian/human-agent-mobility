import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from random import sample
from scipy import linalg
from sklearn.linear_model import LinearRegression
from scipy.stats import gamma
import warnings

#suppress warnings
warnings.filterwarnings('ignore')



# nn=215


## choose the betas that generates the lowest error

def Error_loss_function(Betas_i,X,y):
    
    try:
    
        odds=np.dot(X,Betas_i)
        pi_odds=1/(1+np.exp(odds))
        Loss=(y-pi_odds)**2
        
        return sum(Loss)
    
    except:
        
        return Betas_i
        
        


def choosing_direction(nn,L,H,t1,t2):
    
    
    # nn=10
    mean = [0,1e-1,1e-1,2e-4, 2e-4]
    cov = [[2, 0,0,0,0], [0,(8e-2)**2,0,0,0],[0,0,(8e-2)**2,0,0],[0,0,0,(4e-5)**2,0],[0,0,0,0,(4e-5)**2]] 
    # 
    
    S=250
    
    
    sample_betas=np.random.multivariate_normal(mean, cov, S)
    
    
    file_posterior='input'+ '_'+str(int(nn))+'.csv'
    data_historical=pd.read_csv(file_posterior, header=None)  
    set_observations=data_historical.to_numpy()
    
    
    ## center the inut variables
    
    
    X=np.zeros((set_observations.shape[0],5))
    y=set_observations[:,2]
    X[:,0]=[1]*set_observations.shape[0]
    X[:,1]=set_observations[:,0]-1000
    X[:,2]=set_observations[:,1]-1000
    X[:,3]=X[:,1]**2
    X[:,4]=X[:,2]**2

    
    Losses=[Error_loss_function(sample_betas[i,:],X,y) for i in  range(S)]
    
    
    where_beta_star=np.argmin(Losses)
        
    
    lgth=166
    
    x1=X[-1,1]
    x2=X[-1,2]
    
    comparison_table=np.array([[x1 +lgth,x2,0 ],
                                [x1 +lgth*np.cos(np.pi*0.25), x2+lgth*np.sin(np.pi*0.25),0],
                                [x1, x2 +lgth, 0],
                                [x1-lgth*np.cos(np.pi*0.25), x2+lgth*np.cos(np.pi*0.25),0],
                                [x1-lgth, x2,0],
                                [x1-lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0],
                                [x1, x2-lgth, 0],
                                [x1+lgth*np.cos(np.pi*0.25), x2-lgth*np.cos(np.pi*0.25),0]])
    
    
    
    evaluation_table=np.zeros((8,7))
    
    evaluation_table[:,0]=[1]*8
    evaluation_table[:,1]=comparison_table[:,0]
    evaluation_table[:,2]=comparison_table[:,1]
    evaluation_table[:,3]=comparison_table[:,0]**2
    evaluation_table[:,4]=comparison_table[:,1]**2
    
    
    
    for i in range(8):
        
        evaluation_table[i,5]=np.dot(evaluation_table[i,0:5].T, sample_betas[where_beta_star,:])
    
    
    direction=np.argmin(evaluation_table[:,5])+1
    
    
        
    return (direction,1)
        


# plt.hist(Losses, density=True, bins=30) 



theta=choosing_direction(nn,L,H,t1,t2) 
theta = (float(theta[0]),theta[1])   


