import numpy as np
import pandas as pd
from random import sample
import warnings
from scipy.spatial import distance


#suppress warnings
warnings.filterwarnings('ignore')



# beta1=-0.122
# beta2=-0.122


def Posteriors(X,Y,k,t1,t2,L):
    
    
    k = -8.16e-5
    z_0 = 5
    beta3 = k
    beta4 = k
    
    # Number of observations
    n = X.shape[0] 
    grid_size=6
    
    x_1_hat=np.random.normal(t1, 1000, grid_size)
    x_2_hat=np.random.normal(t2, 1000, grid_size)
    
    
    for i in range(grid_size):
        
        if x_1_hat[i]<0:
            
            x_1_hat[i]=-x_1_hat[i]
            
  
        if  x_1_hat[i]>L:
            
            x_1_hat[i]=2*L -x_1_hat[i]
  
    for i in range(grid_size):
        
        if x_2_hat[i]<0:
        
            x_2_hat[i]= - x_2_hat[i]
        
        if  x_2_hat[i]>L:
            
            x_2_hat[i]=2*L -x_2_hat[i]
  
    x_1_hat=x_1_hat-L/2
    x_2_hat=x_2_hat- L/2
        
    
    # x_1_hat=np.array([np.random.uniform(0,L/3,1)[0],np.random.uniform(L/3,L*(2/3),1)[0],np.random.uniform((L*(0.85)),L,1)[0]])
    # x_2_hat=np.array([np.random.uniform(0,L/3,1)[0],np.random.uniform(L/3,L*(2/3),1)[0],np.random.uniform((L*(0.85)),L,1)[0]])
    
    # x_1_hat=x_1_hat-L/2
    # x_2_hat=x_2_hat-L/2
    
    Beta1=-2*k*x_1_hat
    Beta2=-2*k*x_2_hat

    TableBetas =np.zeros((grid_size,grid_size))
    
    for q in range(grid_size):
        for w in range(grid_size):
            
            beta1=Beta1[q]
            beta2=Beta2[w]
            # print("Beta  1: " +str(beta1))
            # print("Beta  2: " +str(beta2))
            
            TableBetas[q,w]=posterior_update(beta1, beta2, n,X,Y)
            
    TableBetas= TableBetas/np.sum(TableBetas) 
    
    print(TableBetas)
    
    Dict_betas={}
    
    for i in range(grid_size):
        for j in range(grid_size):
            Dict_betas[(i,j)]=TableBetas[i,j]

    try: 
        weights=list(Dict_betas.values())
        indices=np.array(range(grid_size*grid_size))
        sample_weight=np.random.choice(indices, 1, p=weights)[0]
        
        entry_i=int(np.floor(sample_weight/grid_size))
        
        entry_j=sample_weight-entry_i*grid_size
        
        
        beta1_post=Beta1[entry_i]
        beta2_post=Beta2[entry_j]
        
        
        Beta0_posterior=(beta1_post**2)/(4*k) + (beta2_post**2)/(4*k) + z_0
        
    except:
        
        beta1_post=-2*k*t1
        beta2_post=-2*k*t2

        Beta0_posterior=(beta1_post**2)/(4*k) + (beta2_post**2)/(4*k) + z_0
        

    return np.array([Beta0_posterior,beta1_post, beta2_post, beta3, beta4])




def sigmoide(x):   

    return 1/(1 + np.exp(-x))


def posterior_update(beta1, beta2, n,X,Y):
    
    ###kkkkkkkkkk

    k = -8.16e-5
    z_0 = 5

    beta0 = (beta1**2)/(4*k) + (beta2**2)/(4*k) + z_0
    beta3 = k
    beta4 = k

    Betas = np.array([beta0, beta1, beta2, beta3, beta4])

    likelihood_y = 1

    for i in range(n):

        x_i = X[i,:]
        eta = np.dot(x_i, Betas)
        # print("=======================================")
        # print("Probability 0: " +str(1-sigmoide(eta)))
        # print("Probability 1: " +str(sigmoide(eta)))
        # print("Likelihood: " +str((sigmoide(eta)**Y[i])*(1-sigmoide(eta)**(1-Y[i]))))
        likelihood_y*=(sigmoide(eta)**Y[i])*((1-sigmoide(eta))**(1-Y[i]))
        
      
    # likelihood_beta = norm.pdf(beta1, loc=0, scale=25e-2) * norm.pdf(beta2, loc=0, scale=25e-2)
    # likelihood_beta*likelihood_y
    
    return likelihood_y



def choosing_direction(nn,L,H,t1,t2):      
        
    file_posterior='input'+ '_'+str(int(nn))+'.csv'
    data_historical=pd.read_csv(file_posterior, header=None)  
    set_observations=data_historical.to_numpy()
    
    x1=set_observations[-1,0]
    x2=set_observations[-1,1]
    
    ones_array=set_observations[set_observations[:,2]==1]
    zeros_array=set_observations[set_observations[:,2]==0]
    
    
    if ((zeros_array.shape[0])>90):
       
        
       hj=zeros_array.shape[0]
       
       new_zeros_array1=zeros_array[sample(list(range(hj)),30),:]
        
       new_zeros_array2=zeros_array[list(range(-30,0,1)),:]
       
       sub_set_observations1=np.concatenate((new_zeros_array1, new_zeros_array2), axis=0)  
        
       sub_set_observations=np.concatenate((sub_set_observations1, ones_array), axis=0)   
      
        
        
    else:
        
        sub_set_observations=set_observations
    
    
    lgth=168
    
    X=np.zeros((sub_set_observations.shape[0], 5))
    
    
    X[:,0]=[1]*sub_set_observations.shape[0]
    X[:,1]=(sub_set_observations[:,0]-L*0.5)
    X[:,2]=(sub_set_observations[:,1]-L*0.5)
    X[:,3]=(sub_set_observations[:,1]-L*0.5)**2
    X[:,4]=(sub_set_observations[:,0]-L*0.5)**2

    Y=sub_set_observations[:,2]
    
    comparison_table=np.array([[x1 +lgth-L/2,x2-L/2,0 ],
                                [x1 -L/2+lgth*np.cos(np.pi*0.25), x2-L/2+lgth*np.sin(np.pi*0.25),0],
                                [x1-L/2, x2 +lgth-L/2, 0],
                                [x1-lgth*np.cos(np.pi*0.25)-L/2, x2+lgth*np.cos(np.pi*0.25)-L/2,0],
                                [x1-lgth-L/2, x2-L/2,0],
                                [x1-lgth*np.cos(np.pi*0.25)-L/2, x2-lgth*np.cos(np.pi*0.25)-L/2,0],
                                [x1-L/2, x2-lgth-L/2, 0],
                                [x1+lgth*np.cos(np.pi*0.25)-L/2, x2-lgth*np.cos(np.pi*0.25)-L/2,0]])
    
    
    
    
    evaluation_table=np.zeros((8,6))
    
    evaluation_table[:,0]=[1]*8
    evaluation_table[:,1]=comparison_table[:,0]
    evaluation_table[:,2]=comparison_table[:,1]
    evaluation_table[:,3]=comparison_table[:,0]**2
    evaluation_table[:,4]=comparison_table[:,1]**2

    k = -8.16e-5
    
    
    Beta_stars= Posteriors(X,Y,k,t1,t2,L)
    
    if nn>1000:
        
        
        print("================================Atajos")
        
        for i in range(8):
            
            comparison_table[i,2]=distance.euclidean((comparison_table[i,0],comparison_table[i,1]),(t1,t2))
        
        
        which_dir=np.argmin(comparison_table[:,2])
        
        return (which_dir+1, 2)
    
    else:
   
        for i in range(8):
            
            comparison_table[i,2]= sigmoide(np.dot(Beta_stars,evaluation_table[i,0:5]))
        
        
        which_dir=np.argmax(comparison_table[:,2])
        
        return( which_dir+1,2)
        

theta = choosing_direction(nn,L,H,t1,t2)
theta = (float(theta[0]),theta[1])































