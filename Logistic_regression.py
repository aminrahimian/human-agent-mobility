from numpy.linalg import inv
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import norm
# import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
import math


file_prior='Prior_Data.xlsx'
data_prior=pd.read_excel(file_prior, header=0)  
data_prior=data_prior.to_numpy()


post_target=(600,800)


def generate_signal(post_target, pos_agent):
    
    max_radio=250
    # print("max radio: " + str(max_radio))
    diff_x=post_target[0] - pos_agent[0]
    diff_y=post_target[1] - pos_agent[1]
    
    factor=np.sqrt(diff_x**2 +diff_y**2)
    # print("factor : " + str(factor))
    
    p=np.exp(-1*(factor/max_radio))
    # print("proba : " + str(p))
    
    
    return np.random.binomial(1, p)
    

  

class Agent:
    


    
    def __init__(self, x0,y0,data_posterior,epsilon):
        
        self.posx=x0
        self.posy=y0
        self.data_historical=data_posterior
        self.epsilon=epsilon
    
    

    def choosing_direction(self,data_prior):
     
      
        if self.data_historical.shape[0]<data_prior.shape[0]:
            to_sample=data_prior.shape[0]-self.data_historical.shape[0]
            
            sub_sample=random.sample(list(range(to_sample)), to_sample)
            df1=data_prior[sub_sample,:]
        
            complete_data=np.concatenate((df1, self.data_historical), axis=0)
            
            x_mean=np.mean(complete_data[complete_data[:,2]==1,0])
            y_mean=np.mean(complete_data[complete_data[:,2]==1,1])
            
            
            x_col=complete_data[:,0]-x_mean
            y_col=complete_data[:,1]-y_mean
            
            x_col2=x_col**2
            y_col2=y_col**2
            
            r=complete_data[:,2]
            
            useful_data=np.array([x_col,y_col, x_col2,y_col2,r]).T
            
        else:
            
            complete_data=self.data_historical
            
            x_mean=np.mean(complete_data[complete_data[:,2]==1,0])
            y_mean=np.mean(complete_data[complete_data[:,2]==1,1])
            
            
            x_col=complete_data[:,0]-x_mean
            y_col=complete_data[:,1]-y_mean
            
            x_col2=x_col**2
            y_col2=y_col**2
            
            r=complete_data[:,2]
            
            useful_data=np.array([x_col,y_col, x_col2,y_col2,r]).T
            
        
        X=useful_data[:,[0,1,2,3]]
        y=useful_data[:,4]
        
        clf = LogisticRegression(random_state=0,solver='lbfgs', max_iter=500).fit(X, y)
        
        
        up=np.array([[X[-1,:][0], X[-1,:][1]+180,X[-1,:][2],(X[-1,:][1]+180)**2 ]])
        down=np.array([[X[-1,:][0], X[-1,:][1]-180,X[-1,:][2],(X[-1,:][1]-180)**2 ]])
        right=np.array([[X[-1,:][0]+180, X[-1,:][1],(X[-1,:][0]+180)**2,(X[-1,:][3]) ]])
        left=np.array([[X[-1,:][0]-180, X[-1,:][1],(X[-1,:][0]-180)**2,(X[-1,:][3]) ]])
        up_right=np.array([[(X[-1,:][0]+180*np.sin(np.pi*0.25)), (X[-1,:][1]+ 180*np.sin(np.pi*0.25)),(X[-1,:][0]+180*np.sin(np.pi*0.25))**2,(X[-1,:][1]+ 180*np.sin(np.pi*0.25))**2 ]])
        up_left=np.array([[(X[-1,:][0]-180*np.sin(np.pi*0.25)), (X[-1,:][1]+ 180*np.sin(np.pi*0.25)),(X[-1,:][0]-180*np.sin(np.pi*0.25))**2,(X[-1,:][1]+ 180*np.sin(np.pi*0.25))**2 ]])
        down_right=np.array([[(X[-1,:][0]+180*np.sin(np.pi*0.25)), (X[-1,:][1]- 180*np.sin(np.pi*0.25)),(X[-1,:][0]+180*np.sin(np.pi*0.25))**2,(X[-1,:][1]- 180*np.sin(np.pi*0.25))**2 ]])
        down_left=np.array([[(X[-1,:][0]-180*np.sin(np.pi*0.25)), (X[-1,:][1]- 180*np.sin(np.pi*0.25)),(X[-1,:][0]-180*np.sin(np.pi*0.25))**2,(X[-1,:][1]- 180*np.sin(np.pi*0.25))**2 ]])
        
        
        all_dir=np.array([clf.predict_proba(right)[0,1],
                 clf.predict_proba(up_right)[0,1],
                 clf.predict_proba(up)[0,1],
                 clf.predict_proba(up_left)[0,1],
                 clf.predict_proba(left)[0,1],
                 clf.predict_proba(down_left)[0,1],
                 clf.predict_proba(down)[0,1],
                 clf.predict_proba(down_right)[0,1]])
        
        
        positions=np.array([[X[-1,:][0]+180, X[-1,:][1],(X[-1,:][0]+180)**2,(X[-1,:][3])],[(X[-1,:][0]+180*np.sin(np.pi*0.25)), (X[-1,:][1]+ 180*np.sin(np.pi*0.25)),(X[-1,:][0]+180*np.sin(np.pi*0.25))**2,(X[-1,:][1]+ 180*np.sin(np.pi*0.25))**2 ],
                            [X[-1,:][0], X[-1,:][1]+180,X[-1,:][2],(X[-1,:][1]+180)**2 ],[(X[-1,:][0]-180*np.sin(np.pi*0.25)), (X[-1,:][1]+ 180*np.sin(np.pi*0.25)),(X[-1,:][0]-180*np.sin(np.pi*0.25))**2,(X[-1,:][1]+ 180*np.sin(np.pi*0.25))**2 ],
                            [X[-1,:][0]-180, X[-1,:][1],(X[-1,:][0]-180)**2,(X[-1,:][3])], [(X[-1,:][0]-180*np.sin(np.pi*0.25)), (X[-1,:][1]- 180*np.sin(np.pi*0.25)),(X[-1,:][0]-180*np.sin(np.pi*0.25))**2,(X[-1,:][1]- 180*np.sin(np.pi*0.25))**2 ],
                            [X[-1,:][0], X[-1,:][1]-180,X[-1,:][2],(X[-1,:][1]-180)**2 ],[(X[-1,:][0]+180*np.sin(np.pi*0.25)), (X[-1,:][1]- 180*np.sin(np.pi*0.25)),(X[-1,:][0]+180*np.sin(np.pi*0.25))**2,(X[-1,:][1]- 180*np.sin(np.pi*0.25))**2 ]])
            
        
        positions=np.insert(positions, 4, all_dir, axis=1)
        
        if bernoulli.rvs(p=self.epsilon,size=1)[0]==1:
            # print("choses max+++++=============")
            
            indices = np.where(all_dir == all_dir.max())[0][0]
            
            self.posx=positions[indices,0]+x_mean
            self.posy=positions[indices,1]+y_mean
            
            
            
            
            new_row=np.array([[self.posx,self.posy,generate_signal(post_target, ((self.posx, self.posy)))]])
            self.data_historical=np.concatenate((self.data_historical, new_row), axis=0)
            
              
        else:

            indices=random.randint(0,7)   
            
            self.posx=positions[indices,0]+x_mean
            self.posy=positions[indices,1]+y_mean
            
            new_row=np.array([[self.posx,self.posy,generate_signal(post_target, ((self.posx, self.posy)))]])
            self.data_historical=np.concatenate((self.data_historical, new_row), axis=0)
            

epsilon_list=[0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35]

for k in epsilon_list:
    
        
    
    time=[]             
       
    for t in range(200):
    
        
        file_posterior='Input.csv'
        data_posterior=pd.read_csv(file_posterior, header=None)  
        data_posterior=data_posterior.to_numpy()
        
        
        agent1=Agent(0,0,data_posterior,k)
        
        agent1.data_historical
        
        for i in range(1000):
            
            agent1.choosing_direction(data_prior)
            
            
            a=(agent1.posx, agent1.posy)
            d = math.dist(a,post_target)
            
            if d<=50:
                print("==========GAME OVER!==========="+str(i))
                time.append(i)
                break
            
            
    csv_namefile= "data_"+str(k)+".csv"
    np.savetxt(csv_namefile, 
           time,
           delimiter =", ", 
           fmt ='% s')      
    







# agent1.data_historical

# x = agent1.data_historical[:,0]
# y = agent1.data_historical[:,1]

# plt.scatter(x, y)
# plt.plot(x, y)
# plt.scatter(600, 800, c='red')
# plt.show()

# plt.hist(time)



a=(3440,767)
b=(2000,2000)
d = math.dist(a,b)





















