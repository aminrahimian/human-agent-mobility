import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
from scipy.stats import multivariate_normal
import logmab as logmab
import matplotlib.pyplot as plt


post_target=(75,75)

def generate_signal(post_target, pos_agent):
    
    max_radio=np.sqrt(100**2 +100**2)
    # print("max radio: " + str(max_radio))
    diff_x=post_target[0] - pos_agent[0]
    diff_y=post_target[1] - pos_agent[1]
    
    factor=np.sqrt(diff_x**2 +diff_y**2)
    # print("factor : " + str(factor))
    
    p=np.exp(-2*(factor/max_radio))
    # print("proba : " + str(p))
    # return np.random.binomial(1, p)
    
    
    if (p >=0.5):
        
        return 1
    
    else:
        
        return 0
  


class Agent:
    
    
    
    def __init__(self, pos0x, pos0y):
        
        self.posx=pos0x
        self.posy=pos0y
        self.observations=[]
    
    def action_decision(self,dir,delta_t):
        
        reward=0
        
        
        if dir==0:
            
            reward=generate_signal(post_target, (self.posx, self.posy+ delta_t))
            row=[self.posx, self.posy,1,0,0,0, reward]
            self.posy+=delta_t
            
        elif dir==1:
            
            reward=generate_signal(post_target, (self.posx+ delta_t, self.posy))
            row=[self.posx, self.posy,0,1,0,0, reward]
            self.posx+=delta_t
        
        elif dir==2:
          
            reward=generate_signal(post_target, (self.posx, self.posy-delta_t))
            row=[self.posx, self.posy,0,0,1,0, reward]
            self.posy-=delta_t
        
        else:
            
            reward=generate_signal(post_target, (self.posx-delta_t, self.posy))
            row=[self.posx, self.posy,0,0,0,1, reward]
            self.posx-=delta_t
            
        
        return row
    
    
    
    def initial_data(self):
        
        data=np.zeros((20,7))
        
        data[0,:]=self.action_decision(0,60)

        for i in range(1,5):
            
            temp_action=random.choice([0,1,2,3])
            data[i,:]=self.action_decision(temp_action,2)
              
        data[5,:]=self.action_decision(1,60)
        
        for i in range(6,10):
            
            temp_action=random.choice([0,1,2,3])
            data[i,:]=self.action_decision(temp_action,2)
            
        data[10,:]=self.action_decision(2,60)
    
        for i in range(11,15):
            
            temp_action=random.choice([0,1,2,3])
            data[i,:]=self.action_decision(temp_action,2)
            
            
        data[15,:]=self.action_decision(3,60)
          
        for i in range(16,20):
            
            temp_action=random.choice([0,1,2,3])
            data[i,:]=self.action_decision(temp_action,2)
      
          
        
        
        self.observations=data
        
        return data
        

    
    def take_action(self, action,delta_t):
        
        
        if action==np.pi*0.5:
            
            if (self.posy+ delta_t<=100):
                
                reward=generate_signal(post_target, (self.posx, self.posy+ delta_t))
                row_t=[self.posx, self.posy,1,0,0,0, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posy+=delta_t
                
            else:
                
                reward=generate_signal(post_target, (self.posx, self.posy))
                row_t=[self.posx, self.posy,1,0,0,0, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posy+=0
                
        
        elif action==0:
            
            if (self.posx+delta_t <=100):
                reward=generate_signal(post_target, (self.posx+delta_t, self.posy))
                row_t=[self.posx, self.posy,0,1,0,0, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posx+=delta_t
                
            else:
                
                reward=generate_signal(post_target, (self.posx, self.posy))
                row_t=[self.posx, self.posy,0,1,0,0, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posx+=0
            
        
        elif action==np.pi*1.5:
            
            if (self.posy- delta_t>=0):
                
                
                reward=generate_signal(post_target, (self.posx, self.posy- delta_t))
                row_t=[self.posx, self.posy,0,0,1,0, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posy-=delta_t
                
            else:
                
                reward=generate_signal(post_target, (self.posx, self.posy))
                row_t=[self.posx, self.posy,0,0,1,0, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posy-=0
                
            
        else:
            
            if (self.posx-delta_t >=0):
                reward=generate_signal(post_target, (self.posx-delta_t, self.posy))
                row_t=[self.posx, self.posy,0,0,0,1, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posx-=delta_t
            else:
                
                reward=generate_signal(post_target, (self.posx, self.posy))
                row_t=[self.posx, self.posy,0,0,0,1, reward]
                self.observations=np.append(self.observations, [row_t], axis=0)
                self.posx-=0
                




a=Agent(25,25)
data=a.initial_data()
data=pd.DataFrame(data)

new_x=a.posx
new_y=a.posy

new_loc=pd.DataFrame([new_x, new_y])
new_loc=new_loc.T

betas_priors=[0,0,0,0,0,0,0,0]

for t in range(300):
    
    
    direction=logmab.best_action(data,new_loc,betas_priors)[0]
    # betas_priors=logmab.best_action(data,new_loc,betas_priors)[1]
    a.take_action(direction,2)
    a.observations
    
    new_x=a.posx
    new_y=a.posy

    if(((new_x<=80) &(new_x>=70)) & ((new_y<=80) &(new_y>=70))):
        
        break
        
    new_loc=pd.DataFrame([new_x, new_y])
    new_loc=new_loc.T
    
    data=a.observations
    data=pd.DataFrame(data)
   



data.to_csv('py_input.csv', index=False, header=False)
 
new_loc.to_csv('location.csv', index=False, header=False)



generate_signal(post_target, (60,55))

twoDarray=np.zeros((100,100))


for i in range(data.shape[0]):
    
    cor_x=int(data[0][i])
    cor_y=int(data[1][i])
    
    twoDarray[cor_x,cor_y]=twoDarray[cor_x,cor_y]+1
    
    
twoDarray[75,75]=6

plt.figure(figsize=(25,25))
plt.imshow(twoDarray, cmap='binary', interpolation='nearest')
plt.show()











