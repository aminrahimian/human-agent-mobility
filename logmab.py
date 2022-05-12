import random
import numpy as np
import csv
from numpy.linalg import inv
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import norm
from itertools import compress
from scipy.stats import bernoulli
from scipy.stats import uniform
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# data=pd.read_csv('py_input.csv', header=None)  
# new_loc=pd.read_csv('location.csv', header=None)  



def best_action(data,new_loc,betas_priors):
    
        
    y=data[[6]]
    y=list(y[6])
    X=data[range(6)]
    del X[2]
    n=X.shape[0]
    
    intercept=pd.DataFrame({"intercept":pd.Series([1]*n)})
    X = pd.concat([X, intercept], axis=1)
    
    center_X1=X[0].mean()
    center_X2=X[1].mean()
    
    X.loc[:,0]=X[0]-center_X1
    X.loc[:,1]=X[1]-center_X2
    
    X12=X[0]**2
    X22=X[1]**2
    
    X= pd.concat([X, X12,X22], axis=1)
    
    
    m=X.shape[1]
    Betas=[betas_priors]
    sd_prior_betas=[10]*m
    prior_beta=[0]*m
    
    var_pro=0.2*inv(np.dot(X.T, X))
    
    beta=betas_priors
    
    up=[]   
    down=[]
    right=[]
    left=[]
    
    for i in range(1000):
        
        beta_t= list(np.random.multivariate_normal(beta, cov=var_pro))
        
        
        term1=[]
        term2=[]
        
        for j in range(n):
            
            theta_t1=np.exp(np.dot(beta_t, X.loc[j,:]))/(1+(np.exp(np.dot(beta_t, X.loc[j,:]))))
            theta_t2=np.exp(np.dot(beta, X.loc[j,:]))/(1+(np.exp(np.dot(beta, X.loc[j,:]))))
            
            # print("This is theta: " +str(theta_t))
    
            if math.isnan(theta_t1):
                theta_t1=1
                
            term1.append(bernoulli.pmf(int(y[j]),theta_t1))
            term2.append(bernoulli.pmf(int(y[j]),theta_t2))
         
        
        term3=[]
        term4=[]
        
        for j in range(m):
            
            term3.append(norm.pdf(beta_t[j],prior_beta[j],sd_prior_betas[j]))
            term4.append(norm.pdf(beta[j],prior_beta[j],sd_prior_betas[j]))
            
       
        term1=np.log(term1)
        term2=np.log(term2)
        term3=np.log(term3)
        term4=np.log(term4)
        
        
        lhr=sum(term1)-sum(term2) +sum(term3) -sum(term4)
        
        
        treshold=np.log(uniform.rvs(0,1,size=1)[0])
        
        if treshold<lhr:
            
            beta=beta_t
            
           
        act1=1/(1+np.exp(-(np.dot(beta,[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,0,0,0,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
        act2=1/(1+np.exp(-(np.dot(beta,[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,1,0,0,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
        act3=1/(1+np.exp(-(np.dot(beta,[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,0,1,0,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
        act4=1/(1+np.exp(-(np.dot(beta,[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,0,0,1,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
        
        up.append(act1)
        right.append(act2)
        down.append(act3)
        left.append(act4)
        
            
        Betas.append(beta)
                        
    
    # act1=1/(1+np.exp(-(np.dot(Betas[-1],[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,0,0,0,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
    # act2=1/(1+np.exp(-(np.dot(Betas[-2],[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,1,0,0,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
    # act3=1/(1+np.exp(-(np.dot(Betas[-3],[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,0,1,0,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
    # act4=1/(1+np.exp(-(np.dot(Betas[-4],[new_loc[0][0] -center_X1,new_loc[1][0]-center_X2,0,0,1,1,(new_loc[0][0] -center_X1)**2, (new_loc[1][0]-center_X2)**2 ]))))
    
    # act1=1/(1+np.exp(-(np.dot(Betas[-1],[new_loc[0][0],new_loc[1][0],0,0,0,1]))))
    # act2=1/(1+np.exp(-(np.dot(Betas[-2],[new_loc[0][0],new_loc[1][0],1,0,0,1]))))
    # act3=1/(1+np.exp(-(np.dot(Betas[-3],[new_loc[0][0],new_loc[1][0],0,1,0,1]))))
    # act4=1/(1+np.exp(-(np.dot(Betas[-4],[new_loc[0][0],new_loc[1][0],0,0,1,1]))))
    
    
    opt1=random.choice(up)
    opt2=random.choice(right)
    opt3=random.choice(down)
    opt4=random.choice(left)
    
    
    which_act=max(opt1,opt2,opt3,opt4)
    
    if which_act==opt1:
        
        dir=0.5*np.pi
        
    elif which_act==opt2:
        
        dir=0
    
    elif which_act==opt3:
        
        dir=1.5*np.pi
    else:
        
        dir=np.pi
        
        
    
    np.savetxt('action.csv', [dir], delimiter=',') 
        
    return (dir,Betas[-1])
        








def save_beta_hist(k):

    beta1=[Betas[i][k] for i in range(1000)]
    
    plt.hist(beta1, bins=20)
    plt.title("Beta " +str(k+1) )
    plt.ylabel("# Observations")
    plt.show()
    
    # plt.savefig("beta.png")
    
    
    



