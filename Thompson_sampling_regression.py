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

# nn=175


# nn=103


# This function allows to have a sense of the variance in the estimators (Betas). 
# It's useful to establish a set of prior of the variance

def sigma_0_estimation(): 
    
    H=2000   
    L=2000
    
    radius_detection_prior=250
    
    #number of samples to define the prior
    prior_size=int(5)
    
    x_p = np.random.uniform(0,1,prior_size)*L
    y_p = np.random.uniform(0,1,prior_size)*H
    priors= np.zeros((prior_size,3))
    priors[:,0]=x_p
    priors[:,1]=y_p
    
    x_0_p=L*0.5
    y_0_p=H*0.5
    
    
    ## Generate data such that radius of detection is 300 meters
    
    tuple_position=[]
    
    for i in range(prior_size):
        
        tuple_position.append((priors[i,0], priors[i,1]))
    
    
    #function to generate a training set        
    def ones_column(tuple_position):
        
        factor=distance.euclidean((x_0_p,y_0_p),tuple_position)*(2/radius_detection_prior)
        prob=np.exp(-factor)
        return bernoulli.rvs(p=prob,size=1)[0]  
    
    
    
    ones_priors=[ones_column(i) for i in tuple_position]
    priors[:,2]=ones_priors
    
          
    where_ones=priors[priors[:,2]==1]
    where_ones=where_ones[0:40,:]
    where_zeros=priors[priors[:,2]==0]
    where_zeros=where_zeros[0:80,:]
    
    
    priors=np.concatenate((where_ones, where_zeros), axis=0)  
    
    
    ### Be sure that priors contain the information of the historical data
    
    priors_reg= np.zeros((priors.shape[0],5))
    
    priors_reg[:,0]=priors[:,0]-np.mean(priors[:,0])
    priors_reg[:,1]=priors[:,1]-np.mean(priors[:,1])
    priors_reg[:,2]=(priors_reg[:,0])**2
    priors_reg[:,3]=(priors_reg[:,1])**2
    priors_reg[:,4]=priors[:,2]
    
    model = LinearRegression().fit(priors_reg[:,0:4], priors_reg[:,4])
    
    model.coef_
    model.intercept_
    
    y=priors_reg[:,4]
    
    SSR_beta_hat=np.matmul((priors_reg[:,4]-model.predict(priors_reg[:,0:4],)).T, (priors_reg[:,4]-model.predict(priors_reg[:,0:4],)))
    
    sigma_o_est=SSR_beta_hat/(priors_reg.shape[0]-priors_reg.shape[1])
    
    
    return sigma_o_est

# L=2000
# H=2000

    
#### Block of code to generate the actual betas using the actual position of the target


def actual_betas(L,H,t1,t2):
     
    x_p = np.random.uniform(0,1,250)*L
    y_p = np.random.uniform(0,1,250)*H
    simu_data= np.zeros((250,3))
    simu_data[:,0]=x_p
    simu_data[:,1]=y_p
    
    tuple_position=[]
    
    for i in range(250):
        
        tuple_position.append((simu_data[i,0], simu_data[i,1]))
    
    
    
    #function to generate a training set  


    ### CHECK radius_detection_prior

    radius_detection_prior=200
    
    def ones_column(tuple_position):
        
        factor=distance.euclidean((t1,t2),tuple_position)*(1/radius_detection_prior)
        prob=np.exp(-factor)
        return bernoulli.rvs(p=prob,size=1)[0]  
    
    
            
    ones_priors=[ones_column(i) for i in tuple_position]
    simu_data[:,2]=ones_priors
    
    
    
    priors_reg= np.zeros((simu_data.shape[0],5))
    
    priors_reg[:,0]=simu_data[:,0]-np.mean(simu_data[:,0])
    priors_reg[:,1]=simu_data[:,1]-np.mean(simu_data[:,1])
    priors_reg[:,2]=(priors_reg[:,0])**2
    priors_reg[:,3]=(priors_reg[:,1])**2
    priors_reg[:,4]=simu_data[:,2]
    
    model = LinearRegression().fit(priors_reg[:,0:4], priors_reg[:,4])
    
    actual_betas=[model.intercept_]
    actual_betas= actual_betas +(list(model.coef_))
    
    return actual_betas
    
    


def choosing_direction(nn,L,H,t1,t2):
    
      

    file_posterior='input'+ '_'+str(int(nn))+'.csv'
    data_historical=pd.read_csv(file_posterior, header=None)  
    set_observations=data_historical.to_numpy()
    
    m1=int(L/10)
    m2=int(L/5)
    m3=int(L/2.5)
    
    
    
    
    ##First move
    
    if (set_observations.shape[0]<=m1):
        
    
   
        x_0_p=L*0.8
        y_0_p=H*0.2
        
        fixer=np.array([[x_0_p*0.9, y_0_p*1.1,1],
                        [x_0_p*1.01, y_0_p*1.21,1],
                        [x_0_p*1.05, y_0_p*0.85,1]])
        
        
        set_observations=np.concatenate((fixer, set_observations), axis=0)  
        
        radius_detection_prior=250
    
        nu_0=1
        s20=0.12  ## Taken from the above function
        
    
        g=200
        
        ##=====================================
     
        
        
        #number of samples to define the prior
        
        prior_size=int(5)
        
        x_p = np.random.uniform(0,1,prior_size)*L
        y_p = np.random.uniform(0,1,prior_size)*H
        priors= np.zeros((prior_size,3))
        priors[:,0]=x_p
        priors[:,1]=y_p
    
        
        ## Generate data such that radius of detection is 300 meters
        
        tuple_position=[]
        
        for i in range(prior_size):
            
            tuple_position.append((priors[i,0], priors[i,1]))
        
        
        #function to generate a training set        
        def ones_column(tuple_position):
            
            factor=distance.euclidean((x_0_p,y_0_p),tuple_position)*(2/radius_detection_prior)
            prob=np.exp(-factor)
            return bernoulli.rvs(p=prob,size=1)[0]  
        
        
        
        ones_priors=[ones_column(i) for i in tuple_position]
        priors[:,2]=ones_priors
        
        #======================================
        
        set_observations=np.concatenate((priors, set_observations), axis=0)  
        
        
        X=np.zeros((set_observations.shape[0],5))
        y=set_observations[:,2]
        X[:,0]=[1]*set_observations.shape[0]
        X[:,1]=set_observations[:,0]-np.mean(set_observations[:,0])
        X[:,2]=set_observations[:,1]-np.mean(set_observations[:,1])
        X[:,3]=X[:,1]**2
        X[:,4]=X[:,2]**2
        
        
        H_g= (g/(g+1))*np.matmul( X,np.matmul(linalg.inv(np.matmul(X.T,X)),X.T))
        
        SSR_g= np.matmul(y.T,np.matmul(np.eye(set_observations.shape[0])-H_g,y))
        
        
        shape=(nu_0+set_observations.shape[0])*0.5
        scale=((nu_0*s20 + SSR_g)*0.5)**-1
        
        sample_sigma = np.random.gamma(shape, scale, 1)[0]
        
        s21=1/sample_sigma
     
        
     
        Beta_ols=np.matmul(linalg.inv(np.matmul(X.T,X)), np.matmul(X.T,y))
        
        cov_matrix=(g/g+1)*s21*linalg.inv(np.matmul(X.T,X))
        
        
        Betas_s = np.random.multivariate_normal(Beta_ols, cov_matrix, size=1)
    
    
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
            
            evaluation_table[i,5]=np.dot(evaluation_table[i,0:5].T, Betas_s[0,:])
            evaluation_table[i,6]=distance.euclidean((evaluation_table[i,0],evaluation_table[i,1]),(t1,t2))
        
    
        direction=np.argmax(evaluation_table[:,5])+1
        
        
        regret =abs(np.dot(evaluation_table[np.argmax(evaluation_table[:,5]),0:5], np.array(actual_betas(L,H,t1,t2)) )-
                    np.dot(evaluation_table[np.argmin(evaluation_table[:,6]),0:5], np.array(actual_betas(L,H,t1,t2)) ))
        
        return (direction,regret)
    
    # Second move
    
    
    elif((set_observations.shape[0]<=m2) & ((set_observations.shape[0])>m1)):
        
        
        x_0_p=L*0.85
        y_0_p=H*0.85
        
        fixer=np.array([[x_0_p*0.9, y_0_p*1.1,1],
                        [x_0_p*1.01, y_0_p*1.21,1],
                        [x_0_p*1.05, y_0_p*0.85,1],
                        [x_0_p*0.87, y_0_p*1.02,1]])
        
        
        set_observations=np.concatenate((fixer, set_observations), axis=0)  
        
        radius_detection_prior=250
    
        nu_0=1
        s20=0.12  ## Taken from the above function
        
    
        g=200
        
        ##=====================================
     
        
        
        #number of samples to define the prior
        
        prior_size=int(10)
        
        x_p = np.random.uniform(0,1,prior_size)*L
        y_p = np.random.uniform(0,1,prior_size)*H
        priors= np.zeros((prior_size,3))
        priors[:,0]=x_p
        priors[:,1]=y_p
    
        
        ## Generate data such that radius of detection is 300 meters
        
        tuple_position=[]
        
        for i in range(prior_size):
            
            tuple_position.append((priors[i,0], priors[i,1]))
        
        
        #function to generate a training set        
        def ones_column(tuple_position):
            
            factor=distance.euclidean((x_0_p,y_0_p),tuple_position)*(2/radius_detection_prior)
            prob=np.exp(-factor)
            return bernoulli.rvs(p=prob,size=1)[0]  
        
        
        
        ones_priors=[ones_column(i) for i in tuple_position]
        priors[:,2]=ones_priors
        
        #======================================
        
        set_observations=np.concatenate((priors, set_observations), axis=0)  
        
        
        X=np.zeros((set_observations.shape[0],5))
        y=set_observations[:,2]
        X[:,0]=[1]*set_observations.shape[0]
        X[:,1]=set_observations[:,0]-np.mean(set_observations[:,0])
        X[:,2]=set_observations[:,1]-np.mean(set_observations[:,1])
        X[:,3]=X[:,1]**2
        X[:,4]=X[:,2]**2
        
        
        H_g= (g/(g+1))*np.matmul( X,np.matmul(linalg.inv(np.matmul(X.T,X)),X.T))
        
        SSR_g= np.matmul(y.T,np.matmul(np.eye(set_observations.shape[0])-H_g,y))
        
        
        shape=(nu_0+set_observations.shape[0])*0.5
        scale=((nu_0*s20 + SSR_g)*0.5)**-1
        
        sample_sigma = np.random.gamma(shape, scale, 1)[0]
        
        s21=1/sample_sigma
     
        
     
        Beta_ols=np.matmul(linalg.inv(np.matmul(X.T,X)), np.matmul(X.T,y))
        
        cov_matrix=(g/g+1)*s21*linalg.inv(np.matmul(X.T,X))
        
        
        Betas_s = np.random.multivariate_normal(Beta_ols, cov_matrix, size=1)
    
    
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
            
            evaluation_table[i,5]=np.dot(evaluation_table[i,0:5].T, Betas_s[0,:])
            evaluation_table[i,6]=distance.euclidean((evaluation_table[i,0],evaluation_table[i,1]),(t1,t2))
        
    
        direction=np.argmax(evaluation_table[:,5])+1
        
        
        regret =abs(np.dot(evaluation_table[np.argmax(evaluation_table[:,5]),0:5], np.array(actual_betas(L,H,t1,t2)) )-
                    np.dot(evaluation_table[np.argmin(evaluation_table[:,6]),0:5], np.array(actual_betas(L,H,t1,t2)) ))
        
        return (direction,regret)
    
    
    elif((set_observations.shape[0]<=m3) & ((set_observations.shape[0])>m2)):
        
        
        x_0_p=L*0.2
        y_0_p=H*0.5
        
        fixer=np.array([[x_0_p*0.9, y_0_p*1.1,1],
                        [x_0_p*1.01, y_0_p*1.21,1],
                        [x_0_p*1.05, y_0_p*0.85,1],
                        [x_0_p*0.87, y_0_p*1.02,1]])
        
        
        set_observations=np.concatenate((fixer, set_observations), axis=0)  
        
        radius_detection_prior=250
    
        nu_0=1
        s20=0.12  ## Taken from the above function
        
    
        g=200
        
        ##=====================================
     
        
        
        #number of samples to define the prior
        
        prior_size=int(10)
        
        x_p = np.random.uniform(0,1,prior_size)*L
        y_p = np.random.uniform(0,1,prior_size)*H
        priors= np.zeros((prior_size,3))
        priors[:,0]=x_p
        priors[:,1]=y_p
    
        
        ## Generate data such that radius of detection is 300 meters
        
        tuple_position=[]
        
        for i in range(prior_size):
            
            tuple_position.append((priors[i,0], priors[i,1]))
        
        
        #function to generate a training set        
        def ones_column(tuple_position):
            
            factor=distance.euclidean((x_0_p,y_0_p),tuple_position)*(2/radius_detection_prior)
            prob=np.exp(-factor)
            return bernoulli.rvs(p=prob,size=1)[0]  
        
        
        
        ones_priors=[ones_column(i) for i in tuple_position]
        priors[:,2]=ones_priors
        
        #======================================
        
        set_observations=np.concatenate((priors, set_observations), axis=0)  
        
        
        X=np.zeros((set_observations.shape[0],5))
        y=set_observations[:,2]
        X[:,0]=[1]*set_observations.shape[0]
        X[:,1]=set_observations[:,0]-np.mean(set_observations[:,0])
        X[:,2]=set_observations[:,1]-np.mean(set_observations[:,1])
        X[:,3]=X[:,1]**2
        X[:,4]=X[:,2]**2
        
        
        H_g= (g/(g+1))*np.matmul( X,np.matmul(linalg.inv(np.matmul(X.T,X)),X.T))
        
        SSR_g= np.matmul(y.T,np.matmul(np.eye(set_observations.shape[0])-H_g,y))
        
        
        shape=(nu_0+set_observations.shape[0])*0.5
        scale=((nu_0*s20 + SSR_g)*0.5)**-1
        
        sample_sigma = np.random.gamma(shape, scale, 1)[0]
        
        s21=1/sample_sigma
     
        
     
        Beta_ols=np.matmul(linalg.inv(np.matmul(X.T,X)), np.matmul(X.T,y))
        
        cov_matrix=(g/g+1)*s21*linalg.inv(np.matmul(X.T,X))
        
        
        Betas_s = np.random.multivariate_normal(Beta_ols, cov_matrix, size=1)
    
    
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
            
            evaluation_table[i,5]=np.dot(evaluation_table[i,0:5].T, Betas_s[0,:])
            evaluation_table[i,6]=distance.euclidean((evaluation_table[i,0],evaluation_table[i,1]),(t1,t2))
        
    
        direction=np.argmax(evaluation_table[:,5])+1
        
        
        regret =abs(np.dot(evaluation_table[np.argmax(evaluation_table[:,5]),0:5], np.array(actual_betas(L,H,t1,t2)) )-
                    np.dot(evaluation_table[np.argmin(evaluation_table[:,6]),0:5], np.array(actual_betas(L,H,t1,t2)) ))
        
        return (direction,regret)
    
    
    else:
        
        nu_0=1
        s20=0.12  ## Taken from the above function

        g=200
        
        ##====================================
        x_0_p=L*0.5
        y_0_p=H*0.5
        
        fixer=np.array([[x_0_p*0.9, y_0_p*1.1,1],
                        [x_0_p*1.01, y_0_p*1.21,1],
                        [x_0_p*1.05, y_0_p*0.85,1],
                        [x_0_p*0.87, y_0_p*1.02,1]])
        
        
          
        set_observations=set_observations[sample(list(range(set_observations.shape[0])),m1),:]
        
        set_observations=np.concatenate((fixer, set_observations), axis=0)  

        
        X=np.zeros((set_observations.shape[0],5))
        y=set_observations[:,2]
        X[:,0]=[1]*set_observations.shape[0]
        X[:,1]=set_observations[:,0]-np.mean(set_observations[:,0])
        X[:,2]=set_observations[:,1]-np.mean(set_observations[:,1])
        X[:,3]=X[:,1]**2
        X[:,4]=X[:,2]**2
        
        
        H_g= (g/(g+1))*np.matmul( X,np.matmul(linalg.inv(np.matmul(X.T,X)),X.T))
        
        SSR_g= np.matmul(y.T,np.matmul(np.eye(set_observations.shape[0])-H_g,y))
        
        
        shape=(nu_0+set_observations.shape[0])*0.5
        scale=((nu_0*s20 + SSR_g)*0.5)**-1
        sample_sigma = np.random.gamma(shape, scale, 1)[0]
        s21=1/sample_sigma
     
        Beta_ols=np.matmul(linalg.inv(np.matmul(X.T,X)), np.matmul(X.T,y))      
        cov_matrix=(g/g+1)*s21*linalg.inv(np.matmul(X.T,X))
        
        
        Betas_s = np.random.multivariate_normal(Beta_ols, cov_matrix, size=1)
    
    
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
            
            evaluation_table[i,5]=np.dot(evaluation_table[i,0:5].T, Betas_s[0,:])
            evaluation_table[i,6]=distance.euclidean((evaluation_table[i,0],evaluation_table[i,1]),(t1,t2))
        
    
        direction=np.argmax(evaluation_table[:,5])+1
        
        
        regret =abs(np.dot(evaluation_table[np.argmax(evaluation_table[:,5]),0:5], np.array(actual_betas(L,H,t1,t2)) )-
                    np.dot(evaluation_table[np.argmin(evaluation_table[:,6]),0:5], np.array(actual_betas(L,H,t1,t2)) ))
        
        return (direction,regret)
        
        

theta=choosing_direction(nn,L,H,t1,t2) 
theta = (float(theta[0]),theta[1])

















