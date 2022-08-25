import pandas as pd
import numpy as np
from scipy.spatial import distance

def choosing_direction(nn,L,H,t1,t2):
    
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
    
    
    
    
    
    for i in range(8):
        
        comparison_table[i,2]=distance.euclidean((comparison_table[i,0],comparison_table[i,1]),(t1,t2))
    
    
    which_dir=np.argmin(comparison_table[:,2])
    
    return (which_dir+1, 2)
        

theta = choosing_direction(nn,L,H,t1,t2)
theta = (float(theta[0]),theta[1])
   