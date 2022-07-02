import pandas as pd
import numpy as np

def choosing_direction(nn,L,H,t1,t2):
    
    file_posterior='input'+ '_'+str(int(nn))+'.csv'
    data_historical=pd.read_csv(file_posterior, header=None)  
    set_observations=data_historical.to_numpy()
    
    

    
    step_length=168
    
    if ((t1-set_observations[-1,0])>step_length):
        
        
        return 1
    
    else:
        
        if ((t2-set_observations[-1,1])>step_length):
            
            
            return 3
            
        else:
            
            return 3
        
        

theta=choosing_direction(nn,L,H,t1,t2) 
theta = float(theta)
   