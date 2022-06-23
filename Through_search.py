import pandas as pd
import numpy as np
import random



def choosing_direction(nn,L,H):
    
    
    file_posterior='input'+ '_'+str(int(nn))+'.csv'
    data_historical=pd.read_csv(file_posterior, header=None)  
    data_historical=data_historical.to_numpy()
        
    size_space=40
    
    
    x_1=int(np.floor(data_historical[-1,0]/size_space))
    x_2=int(np.floor(data_historical[-1,1]/size_space))
    
        
    if ((x_2%2==0) & ((x_1 >= 1) & (x_1 <=  (int(L/size_space)-2) ))):
        
        ## central zone right.
        
        return 1
    
    elif  ((x_2%2==1) & ((x_1 >= 1) & (x_1 <=  (int(L/size_space)-2) ))):
        
        ##central zone left 
        
        return 5
    
    elif (((x_2%2==0) & (x_2!=(int(L/size_space)-1))) & (x_1==0)):
        
        ##left colum zone right dir
        
        return 1
    
    elif (((x_2%2==1) & (x_2!=(int(L/size_space)-1))) & (x_1==0)):
        
        ##left colum zone up dir
        
        return 3
    
    elif (((x_2%2==0) ) & (x_1== (int(L/size_space)-1))):
        
        ##right colum zone up dir
        
        return 3
    
    elif (((x_2%2==1)) & (x_1== (int(L/size_space)-1))):
        
        ##right colum zone left dir
        
        return 5
    
    elif ((x_2==(int(L/size_space)-1)) &(x_1==0)):
        
        return random.choice([1,8,7])
    